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
st.title("🎯 VCP 다중 타점 계산기 (미너비니식)")

st.markdown("""
**VCP 완성 종목 전용 · 캔들/거래량 기반 타점 분석**
- 정석 VCP(마지막 수축 피벗) / Cheat / Low Cheat / Retest(눌림)
- 타이트 구간 기반 손절 (최대 -10% 제한) | 신뢰도 점수
- 표에서 타점 클릭 → 차트에 진입가/손절가 표시
""")

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
# 타이트 구간 탐지
# -------------------------------------------------
def find_tight_zone(df, lookback=20, max_days=10):
    """마지막 타이트 구간 찾기 (High/Low 모두 반환)"""
    recent = df.tail(lookback)
    atr = recent['ATR20'].iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        z = recent.tail(min(5, len(recent)))
        return z
    
    daily_range = recent['High'] - recent['Low']
    tight_mask = daily_range < (atr * 0.6)
    
    tight_data = recent[tight_mask]
    
    if len(tight_data) == 0:
        return recent.tail(min(5, len(recent)))
    
    return tight_data.tail(max_days)

def cap_stop_by_risk(entry, stop, max_risk=0.10):
    """손절폭이 max_risk를 초과하면 -8%로 강제 제한"""
    if entry <= 0:
        return stop
    risk = (entry - stop) / entry
    if risk > max_risk:
        return entry * 0.92
    return stop

# -------------------------------------------------
# 돌파일 탐지
# -------------------------------------------------
def find_breakout_day(df, pivot_high, lookback=10):
    """피벗 돌파일 찾기: 종가 > 피벗 + 거래량 증가"""
    recent = df.tail(lookback)
    
    for i in range(len(recent)-1, -1, -1):
        row = recent.iloc[i]
        if pd.isna(row['VolAvg60']) or row['VolAvg60'] <= 0:
            continue
            
        # 종가가 피벗 위 + 거래량 1.4배 이상
        if row['Close'] > pivot_high and row['Volume'] >= row['VolAvg60'] * 1.4:
            return i, row
    
    return None, None

# -------------------------------------------------
# 리테스트(눌림) 진입 탐지
# -------------------------------------------------
def find_retest_entry(df, pivot_high, breakout_idx, breakout_row):
    """돌파 후 눌림 진입 조건 체크"""
    if breakout_idx is None:
        return None
    
    recent = df.tail(10)
    breakout_pos = len(recent) - 1 - breakout_idx
    
    # 돌파 후 1~8일 구간 체크
    after_breakout = recent.iloc[breakout_pos+1:]
    
    if len(after_breakout) == 0:
        return None
    
    for i in range(len(after_breakout)):
        row = after_breakout.iloc[i]
        
        # 1) 피벗 근처로 되돌림 (피벗 대비 0~3% 위)
        dist_from_pivot = (row['Close'] - pivot_high) / pivot_high * 100
        if not (0 <= dist_from_pivot <= 3):
            continue
        
        # 2) 거래량 감소 (돌파일 대비)
        if pd.isna(breakout_row['Volume']) or breakout_row['Volume'] <= 0:
            continue
        if row['Volume'] >= breakout_row['Volume'] * 0.8:
            continue
        
        # 3) 타이트한 캔들 (ATR 대비 작은 변동폭)
        if pd.isna(row['ATR20']) or row['ATR20'] <= 0:
            continue
        candle_range = row['High'] - row['Low']
        if candle_range >= row['ATR20'] * 0.6:
            continue
        
        # 4) 반전 신호: 양봉 + 종가가 중간 이상
        body = abs(row['Close'] - row['Open'])
        if row['Close'] <= row['Open']:
            continue
        if (row['Close'] - row['Low']) / candle_range < 0.5:
            continue
        
        # 조건 만족 → 리테스트 진입
        return {
            'entry': float(row['High']),
            'stop': float(row['Low']),
            'date': row.name
        }
    
    return None

# -------------------------------------------------
# 신뢰도 계산
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

def find_low_cheat_trigger(df, lookback=60):
    """Low Cheat 트리거: 강한 양봉 + 거래량"""
    x = df.tail(lookback).copy()
    if len(x) < 30:
        return None

    atr = x["ATR20"]
    vol_avg = x["VolAvg60"]
    body = (x["Close"] - x["Open"]).abs()
    bullish = x["Close"] > x["Open"]

    cond = bullish & (atr > 0) & atr.notna() & (vol_avg > 0) & vol_avg.notna()
    cond &= (body >= 0.6 * atr) & (x["Volume"] >= 1.0 * vol_avg)

    hits = x[cond]
    return df.loc[hits.index[-1]] if len(hits) > 0 else None

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
    elif entry_type == "Retest(눌림)":
        score += 5  # 눌림은 리스크가 작아서 보너스

    return min(int(score), 100)

# -------------------------------------------------
# 타점 계산
# -------------------------------------------------
def calculate_entries(df, atr_buffer_mult=0.3):
    """4가지 진입타점 계산 (Pullback 제거, Retest 추가)"""
    recent = df.tail(120)
    atr20 = recent["ATR20"].iloc[-1]
    
    if pd.isna(atr20) or atr20 <= 0:
        atr20 = recent["Close"].iloc[-1] * 0.02
    
    buffer = atr_buffer_mult * atr20
    
    # 마지막 수축(타이트) 구간
    tz = find_tight_zone(df, lookback=20, max_days=10)
    pivot_high = float(tz["High"].max())
    pivot_low = float(tz["Low"].min())
    
    entries = {}
    
    # 1) 정석 VCP: 마지막 수축 피벗 돌파
    vcp_entry = pivot_high
    vcp_stop = max(100.0, pivot_low - buffer)
    vcp_stop = cap_stop_by_risk(vcp_entry, vcp_stop, max_risk=0.10)
    entries["정석 VCP(피벗)"] = (vcp_entry, vcp_stop)
    
    # 2) Cheat Entry
    base_high = float(recent["High"].max())
    base_low = float(recent["Low"].min())
    base_range = base_high - base_low
    upper_third = base_low + base_range * 0.66
    
    cheat_zone = recent[recent["High"] >= upper_third]
    if len(cheat_zone) > 0:
        cheat_entry = float(cheat_zone["High"].tail(20).max())
        cheat_tight_low = float(cheat_zone['Low'].tail(10).min())
        cheat_stop = max(100.0, cheat_tight_low - buffer)
        cheat_stop = cap_stop_by_risk(cheat_entry, cheat_stop, max_risk=0.10)
    else:
        cheat_entry = base_high * 0.98
        cheat_stop = vcp_stop
    
    entries["Cheat"] = (cheat_entry, cheat_stop)
    
    # 3) Low Cheat
    trigger = find_low_cheat_trigger(df, lookback=60)
    if trigger is not None and not pd.isna(trigger["ATR20"]):
        low_entry = float(trigger["High"])
        low_stop = max(100.0, float(trigger["Low"] - atr_buffer_mult * trigger["ATR20"]))
    else:
        low_entry = float(recent["High"].tail(10).max())
        low_tight_low = pivot_low
        low_stop = max(100.0, low_tight_low - buffer)
    
    low_stop = cap_stop_by_risk(low_entry, low_stop, max_risk=0.10)
    entries["Low Cheat"] = (low_entry, low_stop)
    
    # 4) Retest(눌림): 돌파 후 눌림 진입 (조건부)
    breakout_idx, breakout_row = find_breakout_day(df, pivot_high, lookback=10)
    if breakout_idx is not None:
        retest = find_retest_entry(df, pivot_high, breakout_idx, breakout_row)
        if retest is not None:
            retest_entry = retest['entry']
            retest_stop = max(100.0, retest['stop'] - buffer)
            retest_stop = cap_stop_by_risk(retest_entry, retest_stop, max_risk=0.10)
            entries["Retest(눌림)"] = (retest_entry, retest_stop)
    
    return entries

# -------------------------------------------------
# UI - 상단 입력 영역
# -------------------------------------------------
listing = load_krx_listing()

st.markdown("### 📥 입력")
col1, col2, col3 = st.columns([3, 2, 3])

with col1:
    user_input = st.text_input(
        "종목 코드 또는 종목명",
        placeholder="예: 005930 또는 삼성전자",
        help="코드(6자리) 또는 종목명(부분일치) 입력"
    )

with col2:
    atr_buffer_mult = st.slider("ATR 버퍼 배수", 0.1, 1.0, 0.3, 0.1)

with col3:
    with st.expander("💡 타점 설명"):
        st.markdown("""
**정석 VCP(피벗)**: 마지막 수축 피벗 돌파 | 타이트 저점 스탑  
**Cheat**: 상단 1/3 고점 진입 | 상단 타이트 저점 스탑  
**Low Cheat**: 트리거 바 고가 진입 | 트리거 바 저점 스탑  
**Retest(눌림)**: 돌파 후 피벗 리테스트 | 눌림 저점 스탑  
※ 리스크 -10% 초과 시 자동 -8% 조정  
※ Retest는 돌파일 확인 + 거래량 감소 + 반전 캔들 조건 만족 시에만 표시
""")

st.divider()

# Session state 초기화
if 'selected_entry_idx' not in st.session_state:
    st.session_state.selected_entry_idx = 0

# -------------------------------------------------
# 하단 결과 영역
# -------------------------------------------------
if not user_input:
    st.info("👆 종목 코드(6자리) 또는 종목명을 입력하세요")
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
            
            # 현재가 표시
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔹 현재가", f"{current_price:,.0f}원", 
                     delta=f"{((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
            
            atr20 = df["ATR20"].iloc[-1]
            if not pd.isna(atr20):
                atr_pct = atr20 / current_price * 100
                m2.metric("ATR(20)", f"{atr20:,.0f}원")
                m3.metric("ATR / 현재가", f"{atr_pct:.2f}%")
            
            entries = calculate_entries(df, atr_buffer_mult=atr_buffer_mult)

            if len(entries) == 0:
                st.warning("⚠️ 조건을 만족하는 진입 타점이 없습니다.")
            else:
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

                st.markdown("### 📊 타점 비교 (신뢰도 순) - 클릭하여 차트에 표시")
                
                # 테이블 표시 (클릭 이벤트)
                display = df_result.copy()
                display["진입가"] = display["진입가"].map(lambda x: f"{x:,.0f}")
                display["손절가"] = display["손절가"].map(lambda x: f"{x:,.0f}")
                display["R(원)"] = display["R(원)"].map(lambda x: f"{x:,.0f}")
                display["손절폭(%)"] = display["손절폭(%)"].map(lambda x: f"{x:.1f}%")
                display["현재가 대비(%)"] = display["현재가 대비(%)"].map(lambda x: f"{x:+.1f}%")

                event = st.dataframe(
                    display[["순위","타점","진입가","손절가","R(원)","손절폭(%)","현재가 대비(%)","신뢰도"]],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )

                # 선택된 행 처리
                if event.selection.rows:
                    st.session_state.selected_entry_idx = event.selection.rows[0]
                
                selected_idx = st.session_state.selected_entry_idx
                selected_idx = max(0, min(selected_idx, len(df_result) - 1))
                
                selected_row = df_result.iloc[selected_idx]
                selected_entry = selected_row["진입가"]
                selected_stop = selected_row["손절가"]
                selected_name = selected_row["타점"]
                
                dist_pct = selected_row['현재가 대비(%)']
                
                col_msg1, col_msg2 = st.columns(2)
                with col_msg1:
                    st.info(f"""🎯 **선택된 타점**: {selected_name} (순위: {selected_row['순위']})
- 신뢰도: {selected_row['_score']}점 | 진입가: {selected_entry:,.0f}원
- 손절가: {selected_stop:,.0f}원 | R: {selected_row['R(원)']:,.0f}원
- 손절폭: {selected_row['손절폭(%)']:.1f}% | 현재가 대비: {selected_row['현재가 대비(%)']:+.1f}%
""")
                
                with col_msg2:
                    if dist_pct < -3:
                        st.warning(f"⚠️ 이미 돌파됨 (현재가: {current_price:,.0f}원)")
                    elif dist_pct > 10:
                        st.info(f"💡 진입가까지 {dist_pct:.1f}% 떨어져 있음")
                    else:
                        st.success(f"✅ 진입 대기 구간 ({dist_pct:+.1f}%)")

                st.divider()
                st.markdown(f"### 📈 차트 - {selected_name} (진입: 초록 | 손절: 빨강)")
                
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
                    line=dict(color="blue", dash="dot", width=1)
                ))

                # 현재가 라인 (주황색)
                fig.add_trace(go.Scatter(
                    x=[chart_df.index[0], chart_df.index[-1]],
                    y=[current_price, current_price],
                    name=f"현재가 ({current_price:,.0f})",
                    line=dict(color="orange", dash="solid", width=2)
                ))

                # 선택된 타점의 진입가 라인 (초록색)
                fig.add_trace(go.Scatter(
                    x=[chart_df.index[0], chart_df.index[-1]],
                    y=[selected_entry, selected_entry],
                    name=f"진입가 - {selected_name} ({selected_entry:,.0f})",
                    line=dict(color="green", dash="dash", width=2.5)
                ))

                # 선택된 타점의 손절가 라인 (빨강색)
                fig.add_trace(go.Scatter(
                    x=[chart_df.index[0], chart_df.index[-1]],
                    y=[selected_stop, selected_stop],
                    name=f"손절가 ({selected_stop:,.0f})",
                    line=dict(color="red", dash="dash", width=2.5)
                ))

                fig.update_layout(
                    height=600,
                    title=f"{name+' ' if name else ''}{code} | {selected_name} | 현재: {current_price:,.0f} | 진입: {selected_entry:,.0f} | 손절: {selected_stop:,.0f}",
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("✅ 정석 VCP는 마지막 수축 피벗 기준 | Retest(눌림)는 돌파 후 거래량 감소+반전 캔들 조건 만족 시에만 표시 | 손절가는 타이트 구간 저점 기반 (-10% 초과 시 자동 -8% 조정)")


