import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="VCP Multi-Entry Calculator", layout="wide")
st.title("🎯 VCP 다중 타점 계산기 (미너비니식 구조 스탑)")

st.markdown("""
**VCP 완성 종목 전용 · 4가지 타점 자동 분석**

- 정석 VCP / Cheat / Low Cheat / Pullback
- 타점별 Entry · Stop · R 자동 계산
- **모든 타점: 구조 기반 손절 (ATR 버퍼 적용)**
- 신뢰도 점수 (같은 종목 내 비교용)
""")

st.caption("※ 모든 손절가는 '구조적 무효화 지점(스윙/타이트 구간 저점) - ATR 버퍼'로 계산됩니다.")

# -------------------------------------------------
# 종목명/코드 매핑
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_krx_listing():
    """KRX 종목 리스트"""
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq], ignore_index=True)
        
        if 'Symbol' in stocks.columns:
            stocks = stocks.rename(columns={'Symbol': 'Code'})
        elif 'code' in stocks.columns:
            stocks = stocks.rename(columns={'code': 'Code'})
        
        stocks['Code'] = stocks['Code'].astype(str).str.zfill(6)
        return stocks[['Code', 'Name']].dropna().drop_duplicates()
    except:
        try:
            backup = pd.read_csv('krx_backup.csv')
            if 'Symbol' in backup.columns:
                backup = backup.rename(columns={'Symbol': 'Code'})
            elif 'code' in backup.columns:
                backup = backup.rename(columns={'code': 'Code'})
            backup['Code'] = backup['Code'].astype(str).str.zfill(6)
            return backup[['Code', 'Name']].dropna().drop_duplicates()
        except:
            return pd.DataFrame(columns=['Code', 'Name'])

def resolve_code(user_input: str, listing: pd.DataFrame):
    """종목코드/종목명 → Code 변환"""
    s = (user_input or "").strip()
    if not s:
        return None, None

    if s.isdigit():
        code = s.zfill(6)
        m = listing[listing["Code"] == code]
        name = m.iloc[0]["Name"] if len(m) > 0 else None
        return code, name

    hits = listing[listing["Name"].str.contains(s, case=False, na=False)]
    if len(hits) == 0:
        return None, None
    if len(hits) == 1:
        return hits.iloc[0]["Code"], hits.iloc[0]["Name"]

    options = [f"{r.Name} ({r.Code})" for r in hits.itertuples(index=False)]
    picked = st.selectbox("동일/유사 종목명이 여러 개입니다. 선택하세요.", options)
    code = picked.split("(")[-1].replace(")", "").strip()
    name = picked.split("(")[0].strip()
    return code, name

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(code):
    """주식 데이터 로딩"""
    end = datetime.now()
    start = end - timedelta(days=260)
    try:
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 120 else None
    except:
        return None

# -------------------------------------------------
# 지표 계산
# -------------------------------------------------
def prepare_indicators(df):
    """기술적 지표 + ATR 계산"""
    df = df.copy()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["VolAvg60"] = df["Volume"].rolling(60).mean()

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR20"] = tr.rolling(20).mean()
    return df

# -------------------------------------------------
# 거래량 Dry-up 점수
# -------------------------------------------------
def volume_dry_score(df):
    """거래량 고갈 정도 (0.6 ~ 1.0)"""
    recent_min = df["Volume"].tail(3).min()
    avg60 = df["VolAvg60"].iloc[-1]
    if pd.isna(avg60) or avg60 == 0:
        return 0.6

    ratio = recent_min / avg60
    if ratio < 0.4:
        return 1.0
    elif ratio < 0.6:
        return 0.8
    return 0.6

# -------------------------------------------------
# 거리 가중치
# -------------------------------------------------
def distance_weight(entry, current):
    """현재가 vs 진입가 거리 가중"""
    if entry == 0:
        return 0.5
    dist_pct = ((entry - current) / current) * 100
    if dist_pct < -3:
        return 0.5
    if 0 <= dist_pct < 2:
        return 1.0
    if 2 <= dist_pct < 5:
        return 0.95
    if 5 <= dist_pct < 8:
        return 0.85
    return 0.7

# -------------------------------------------------
# Low Cheat 트리거 탐지
# -------------------------------------------------
def find_low_cheat_trigger(df, lookback=60):
    """Low Cheat 트리거: 강한 양봉 + 거래량"""
    x = df.tail(lookback).copy()
    if len(x) < 30:
        return None

    atr = x["ATR20"]
    vol_avg = x["VolAvg60"]
    body = (x["Close"] - x["Open"]).abs()
    bullish = x["Close"] > x["Open"]

    cond = bullish
    cond &= atr.notna() & (atr > 0)
    cond &= vol_avg.notna() & (vol_avg > 0)
    cond &= (body >= 0.6 * atr)
    cond &= (x["Volume"] >= 1.0 * vol_avg)

    hits = x[cond]
    if len(hits) == 0:
        return None

    return df.loc[hits.index[-1]]

# -------------------------------------------------
# 타점 계산 (미너비니식 구조 스탑)
# -------------------------------------------------
def calculate_entries(df, atr_buffer_mult=0.3):
    """4가지 진입타점 계산 (모두 구조 기반 손절 + ATR 버퍼)"""
    recent = df.tail(120)
    atr20 = recent["ATR20"].iloc[-1]
    
    # ATR 없으면 fallback
    if pd.isna(atr20) or atr20 <= 0:
        atr20 = recent["Close"].iloc[-1] * 0.02
    
    buffer = atr_buffer_mult * atr20
    
    base_high = float(recent["High"].max())
    base_low = float(recent["Low"].min())
    base_range = base_high - base_low
    upper_third = base_low + base_range * 0.66
    
    # 1) 정석 VCP
    vcp_entry = base_high
    tight_zone = recent.tail(20)
    vcp_structure_low = float(tight_zone["Low"].min())
    vcp_stop = max(100.0, vcp_structure_low - buffer)
    
    # 2) Cheat Entry
    cheat_zone = recent[recent["High"] >= upper_third]
    if len(cheat_zone) > 0:
        cheat_entry = float(cheat_zone["High"].tail(20).max())
        cheat_structure_low = float(cheat_zone["Low"].min())
        cheat_stop = max(100.0, cheat_structure_low - buffer)
    else:
        cheat_entry = base_high * 0.98
        cheat_stop = max(100.0, vcp_structure_low - buffer)
    
    # 3) Low Cheat
    trigger = find_low_cheat_trigger(df, lookback=60)
    if trigger is not None and not pd.isna(trigger["ATR20"]):
        low_entry = float(trigger["High"])
        low_stop = max(100.0, float(trigger["Low"] - atr_buffer_mult * trigger["ATR20"]))
    else:
        low_entry = float(recent["High"].tail(10).max())
        low_stop = max(100.0, float(recent["Low"].tail(10).min() - buffer))
    
    # 4) Pullback
    pull_entry = base_high
    pullback_zone = recent.tail(10)
    pull_structure_low = float(pullback_zone["Low"].min())
    pull_stop = max(100.0, pull_structure_low - buffer)
    
    return {
        "정석 VCP": (vcp_entry, vcp_stop),
        "Cheat": (cheat_entry, cheat_stop),
        "Low Cheat": (low_entry, low_stop),
        "Pullback": (pull_entry, pull_stop),
    }

# -------------------------------------------------
# 신뢰 점수
# -------------------------------------------------
def confidence_score(entry, stop, df, entry_type):
    """타점 신뢰도 (0~100)"""
    current = df["Close"].iloc[-1]
    r = entry - stop
    if r <= 0:
        return 0

    score = 50.0
    score += volume_dry_score(df) * 25
    score += distance_weight(entry, current) * 15

    short_range = (df["High"].tail(10) - df["Low"].tail(10)).mean()
    long_range = (df["High"].tail(60) - df["Low"].tail(60)).mean()
    if long_range > 0 and short_range / long_range < 0.6:
        score += 10

    risk_pct = (entry - stop) / entry

    if entry_type == "Low Cheat":
        if risk_pct > 0.07:
            score -= 12
        if abs(entry - current) / current < 0.03:
            score += 5
    elif entry_type == "Cheat":
        score *= 0.95
    elif entry_type == "Pullback" and current < entry:
        score *= 0.8

    return min(int(score), 100)

# -------------------------------------------------
# UI
# -------------------------------------------------
listing = load_krx_listing()

col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📥 입력")

    user_input = st.text_input(
        "종목 코드 또는 종목명",
        placeholder="예: 005930 또는 삼성전자",
        help="코드(6자리) 또는 종목명(부분일치) 입력"
    )

    atr_buffer_mult = st.slider("ATR 버퍼 배수 (모든 타점 공통)", 0.1, 1.0, 0.3, 0.1)
    st.caption("손절 = 구조 저점 - (ATR × 버퍼)")

    st.divider()

    with st.expander("💡 타점 설명 (미너비니식)"):
        st.markdown("""
**정석 VCP**
- Entry: 베이스 최고가
- Stop: 마지막 타이트 구간(20일) 저점 - ATR버퍼

**Cheat**
- Entry: 베이스 상단 1/3 고점
- Stop: 상단 1/3 구간 저점 - ATR버퍼

**Low Cheat**
- Entry: 트리거 바(강한 양봉) 고가
- Stop: 트리거 바 저점 - ATR버퍼

**Pullback**
- Entry: 베이스 최고가 (재테스트)
- Stop: 풀백 구간(10일) 저점 - ATR버퍼
""")

with col_output:
    if not user_input:
        st.info("👈 종목 코드(6자리) 또는 종목명을 입력하세요")
    else:
        code, name = resolve_code(user_input, listing)

        if not code:
            st.error("❌ 종목을 찾지 못했습니다.")
        else:
            if name:
                st.subheader(f"📌 {name} ({code})")
            else:
                st.subheader(f"📌 {code}")

            df = load_data(code)
            if df is None:
                st.error("❌ 데이터 로딩 실패")
            else:
                df = prepare_indicators(df)
                current_price = float(df["Close"].iloc[-1])
                
                st.metric("🔹 현재가", f"{current_price:,.0f}원", 
                         delta=f"{((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
                
                entries = calculate_entries(df, atr_buffer_mult=atr_buffer_mult)

                rows = []
                for entry_name, (entry, stop) in entries.items():
                    score = confidence_score(entry, stop, df, entry_name)
                    r_value = entry - stop
                    dist_from_current = ((entry - current_price) / current_price) * 100
                    
                    rows.append({
                        "타점": entry_name,
                        "진입가": float(entry),
                        "손절가": float(stop),
                        "R(원)": float(r_value),
                        "손절폭(%)": float((stop - entry) / entry * 100),
                        "현재가 대비(%)": float(dist_from_current),
                        "신뢰도": int(score),
                        "_score": int(score),
                    })

                df_result = pd.DataFrame(rows).sort_values("_score", ascending=False).reset_index(drop=True)
                df_result.insert(0, "순위", range(1, len(df_result) + 1))

                st.subheader("📊 타점 비교 (신뢰도 순)")
                display = df_result.copy()
                display["진입가"] = display["진입가"].map(lambda x: f"{x:,.0f}")
                display["손절가"] = display["손절가"].map(lambda x: f"{x:,.0f}")
                display["R(원)"] = display["R(원)"].map(lambda x: f"{x:,.0f}")
                display["손절폭(%)"] = display["손절폭(%)"].map(lambda x: f"{x:.1f}%")
                display["현재가 대비(%)"] = display["현재가 대비(%)"].map(lambda x: f"{x:+.1f}%")

                st.dataframe(
                    display[["순위","타점","진입가","손절가","R(원)","손절폭(%)","현재가 대비(%)","신뢰도"]],
                    use_container_width=True,
                    hide_index=True
                )

                best = df_result.iloc[0]
                st.success(f"""⭐ **자동 추천 타점**: {best['타점']}
- 신뢰도: {best['_score']}점
- 진입가: {best['진입가']:,.0f}원
- 손절가: {best['손절가']:,.0f}원
- R: {best['R(원)']:,.0f}원
- 손절폭: {best['손절폭(%)']:.1f}%
- 현재가 대비: {best['현재가 대비(%)']:+.1f}%
""")

                dist_pct = best['현재가 대비(%)']
                if dist_pct < -3:
                    st.warning(f"⚠️ 이미 돌파됨 (현재가: {current_price:,.0f}원)")
                elif dist_pct > 10:
                    st.info(f"💡 진입가까지 {dist_pct:.1f}% 떨어져 있음")
                else:
                    st.success(f"✅ 진입 대기 구간 ({dist_pct:+.1f}%)")

                st.divider()
                st.markdown("### 📐 변동성 (ATR 20일)")
                atr20 = df["ATR20"].iloc[-1]
                if not pd.isna(atr20):
                    atr_pct = atr20 / current_price * 100
                    col1, col2, col3 = st.columns(3)
                    col1.metric("현재가", f"{current_price:,.0f}원")
                    col2.metric("ATR(20)", f"{atr20:,.0f}원")
                    col3.metric("ATR / 현재가", f"{atr_pct:.2f}%")
                else:
                    st.warning("ATR 계산 불가")

                st.divider()
                st.markdown("### 📈 차트")
                fig = go.Figure()
                chart_df = df.tail(120)

                fig.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["Open"],
                    high=chart_df["High"],
                    low=chart_df["Low"],
                    close=chart_df["Close"],
                    name="Price"
                ))

                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df["MA50"],
                    name="50MA",
                    line=dict(color="blue", dash="dot")
                ))

                fig.add_trace(go.Scatter(
                    x=[chart_df.index[0], chart_df.index[-1]],
                    y=[current_price, current_price],
                    name=f"현재가 ({current_price:,.0f})",
                    line=dict(color="orange", dash="solid", width=2)
                ))

                fig.update_layout(
                    height=600,
                    title=f"{name+' ' if name else ''}{code} (현재가: {current_price:,.0f}원)",
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("✅ 모든 손절가는 구조 기반(스윙/타이트 구간 저점 - ATR 버퍼)으로 계산됩니다.")


