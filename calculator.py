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
st.title("🎯 VCP 다중 타점 계산기")

st.markdown("""
**VCP 완성 종목 전용 · 4가지 타점 자동 분석**

- 정석 VCP / Cheat / Low Cheat / Pullback
- 타점별 Entry · Stop · R 자동 계산
- 신뢰도 점수 (같은 종목 내 비교용)
- ✅ 20일 ATR (변동성 참고용)
""")

st.caption("※ ATR은 참고용 정보이며, 버퍼(스탑 여유폭) 계산에만 사용됩니다.")

# -------------------------------------------------
# 종목명/코드 매핑
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_krx_listing():
    # KRX 전체 상장 목록 (Name, Symbol/Code 컬럼이 버전마다 다를 수 있어 방어)
    df = fdr.StockListing("KRX")
    df = df.rename(columns={
        "Symbol": "Code",
        "code": "Code",
        "종목코드": "Code",
        "Name": "Name",
        "종목명": "Name",
        "Market": "Market",
        "시장": "Market",
    })
    if "Code" not in df.columns:
        # 일부 환경에서 Code가 다른 이름일 수 있어 최소한의 fallback
        possible = [c for c in df.columns if c.lower() in ("symbol", "code", "short_code")]
        if possible:
            df = df.rename(columns={possible[0]: "Code"})
    if "Name" not in df.columns:
        possible = [c for c in df.columns if c.lower() in ("name", "codename")]
        if possible:
            df = df.rename(columns={possible[0]: "Name"})
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df[["Code", "Name"]].dropna().drop_duplicates()

def resolve_code(user_input: str, listing: pd.DataFrame):
    """사용자 입력이 코드(6자리)면 그대로, 아니면 종목명 부분일치로 Code 반환"""
    s = (user_input or "").strip()
    if not s:
        return None, None

    # 6자리 숫자면 코드로 간주
    if s.isdigit():
        code = s.zfill(6)
        name = None
        m = listing[listing["Code"] == code]
        if len(m) > 0:
            name = m.iloc[0]["Name"]
        return code, name

    # 종목명 검색 (부분일치)
    hits = listing[listing["Name"].str.contains(s, case=False, na=False)]
    if len(hits) == 0:
        return None, None
    if len(hits) == 1:
        return hits.iloc[0]["Code"], hits.iloc[0]["Name"]

    # 여러 개면 선택
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
        df = fdr.DataReader(code, start, end)  # KRX는 6자리 코드 사용 [web:156]
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
# Low Cheat 트리거 탐지 (자동화 버전)
# -------------------------------------------------
def find_low_cheat_trigger(df, lookback=60):
    """
    Low Cheat 트리거(자동 근사):
    - 최근 lookback 내에서
    - 양봉(종가 > 시가)
    - 바디가 ATR 대비 어느 정도 있고(캐릭터 체인지 근사)
    - 거래량이 60일 평균 이상
    중 "최근"에 해당하는 바를 트리거로 선택
    """
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
    cond &= (body >= 0.6 * atr)          # 트리거 바디 최소 조건(근사)
    cond &= (x["Volume"] >= 1.0 * vol_avg)

    hits = x[cond]
    if len(hits) == 0:
        return None

    # 가장 최근 트리거를 사용
    trigger_idx = hits.index[-1]
    return df.loc[trigger_idx]

# -------------------------------------------------
# 타점 계산
# -------------------------------------------------
def calculate_entries(df, atr_buffer_mult=0.3):
    """4가지 타점 계산 (Low Cheat은 ATR 기반 버퍼 적용)"""
    recent = df.tail(120)

    base_high = recent["High"].max()
    base_low = recent["Low"].min()
    base_range = base_high - base_low
    upper_third = base_low + base_range * 0.66

    # 1) 정석 VCP (기존 로직 유지: entry=base_high)
    vcp_entry = base_high
    vcp_stop = base_high * 0.95  # TODO: 구조 기반으로 바꾸고 싶으면 다음 단계에서 변경

    # 2) Cheat Entry (기존 로직 유지)
    cheat_zone = recent[recent["High"] >= upper_third]
    cheat_entry = cheat_zone["High"].tail(20).max() if len(cheat_zone) else base_high * 0.98
    cheat_stop = cheat_entry * 0.96

    # 3) Low Cheat (미너비니식에 가깝게: 트리거 고가 돌파 / 트리거 저가 - ATR버퍼)
    trigger = find_low_cheat_trigger(df, lookback=60)
    if trigger is not None and not pd.isna(trigger["ATR20"]):
        low_cheat_entry = float(trigger["High"])
        low_cheat_stop = float(trigger["Low"] - atr_buffer_mult * trigger["ATR20"])
    else:
        # 트리거가 안 잡히면 fallback (표시/사용은 되지만 신뢰도는 낮게 나올 것)
        low_cheat_entry = float(recent["High"].tail(10).max())
        ma50 = float(recent["MA50"].iloc[-1]) if not pd.isna(recent["MA50"].iloc[-1]) else float(recent["Low"].tail(10).min())
        atr20 = recent["ATR20"].iloc[-1]
        buffer = float(atr_buffer_mult * atr20) if not pd.isna(atr20) else 0.0
        low_cheat_stop = float(max(ma50, recent["Low"].tail(10).min()) - buffer)

    # 4) Pullback (기존 로직 유지)
    pullback_entry = base_high
    pullback_stop = base_high * 0.97

    return {
        "정석 VCP": (vcp_entry, vcp_stop),
        "Cheat": (cheat_entry, cheat_stop),
        "Low Cheat": (low_cheat_entry, low_cheat_stop),
        "Pullback": (pullback_entry, pullback_stop)
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
        # Low Cheat은 원래 리스크가 짧아야 하므로, 너무 넓으면 벌점
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

    atr_buffer_mult = st.slider("Low Cheat ATR 버퍼 배수", 0.1, 1.0, 0.3, 0.1)
    st.caption("예: 0.3이면 손절 = 트리거 저가 - 0.3×ATR20")

    st.divider()

    with st.expander("💡 타점 설명(현재 로직 기준)"):
        st.markdown("""
**정석 VCP**
- Entry: 최근 120일 베이스 최고가
- Stop: Entry -5% (현재는 고정, 다음 단계에서 구조 기반으로 변경 가능)

**Cheat**
- Entry: 베이스 상단 1/3 영역에서 최근 고점(근사)
- Stop: Entry -4% (현재는 고정)

**Low Cheat (개선)**
- Entry: 최근 60일 내 트리거 바(강한 양봉+거래량) 고가 돌파
- Stop: 트리거 바 저가 - (ATR 버퍼)

**Pullback**
- Entry: 베이스 최고가(돌파 후 리테스트 가정)
- Stop: Entry -3% (현재는 고정)
""")

with col_output:
    if not user_input:
        st.info("👈 종목 코드(6자리) 또는 종목명을 입력하세요")
    else:
        code, name = resolve_code(user_input, listing)

        if not code:
            st.error("❌ 종목을 찾지 못했습니다. (코드/종목명 확인)")
        else:
            if name:
                st.subheader(f"📌 선택 종목: {name} ({code})")
            else:
                st.subheader(f"📌 선택 종목: {code}")

            df = load_data(code)
            if df is None:
                st.error("❌ 데이터 로딩 실패")
            else:
                df = prepare_indicators(df)
                entries = calculate_entries(df, atr_buffer_mult=atr_buffer_mult)

                rows = []
                for entry_name, (entry, stop) in entries.items():
                    score = confidence_score(entry, stop, df, entry_name)
                    r_value = entry - stop
                    rows.append({
                        "타점": entry_name,
                        "진입가": float(entry),
                        "손절가": float(stop),
                        "R(원)": float(r_value),
                        "손절폭(%)": float((stop - entry) / entry * 100),
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

                st.dataframe(
                    display[["순위","타점","진입가","손절가","R(원)","손절폭(%)","신뢰도"]],
                    use_container_width=True,
                    hide_index=True
                )

                best = df_result.iloc[0]
                st.success(f"""⭐ **자동 추천 타점**: {best['타점']}
- 신뢰도: {best['_score']}점
- 진입가: {best['진입가']:,.0f}
- 손절가: {best['손절가']:,.0f}
- R: {best['R(원)']:,.0f}
- 손절폭: {best['손절폭(%)']:.1f}%
""")

                current_price = df["Close"].iloc[-1]
                recommended_entry = best["진입가"]
                dist_pct = ((recommended_entry - current_price) / current_price) * 100

                if dist_pct < -3:
                    st.warning(f"⚠️ 이미 돌파됨 (현재가: {current_price:,.0f})")
                elif dist_pct > 10:
                    st.info(f"💡 진입가까지 {dist_pct:.1f}% 떨어져 있음 (현재가: {current_price:,.0f})")
                else:
                    st.success(f"✅ 진입 대기 구간 (현재가: {current_price:,.0f}, {dist_pct:+.1f}%)")

                st.divider()
                st.markdown("### 📐 변동성 (ATR 20일)")
                atr20 = df["ATR20"].iloc[-1]
                if not pd.isna(atr20):
                    atr_pct = atr20 / current_price * 100
                    col1, col2 = st.columns(2)
                    col1.metric("ATR(20)", f"{atr20:,.0f}원")
                    col2.metric("ATR / 현재가", f"{atr_pct:.2f}%")
                else:
                    st.warning("ATR 계산 불가")

                # 차트 (수평선 제거: 캔들 + 50MA만)
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

                fig.update_layout(
                    height=600,
                    title=f"{name+' ' if name else ''}{code} - VCP 다중 타점",
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("※ ATR은 변동성 참고/버퍼용이며, 매매 기준은 구조가 우선입니다.")

