import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from difflib import get_close_matches

# -----------------------------
# 페이지
# -----------------------------
st.set_page_config(page_title="Entry Confidence (KR)", layout="wide")
st.title("📌 진입 신뢰도 평가기 (종가 확정 기준)")

st.markdown("""
- 종목 입력 → 상황(셋업) 선택 → **내 진입가** 입력 → 체크리스트/신뢰도/손절 후보를 즉시 계산
- 손절폭이 **-8% 초과면 FAIL(진입불가)** 처리 (오닐식 리스크 관리)  
""")

# -----------------------------
# 종목 리스트 로딩 (캐시)
# -----------------------------
@st.cache_data(ttl=86400, show_spinner="종목 리스트 로딩 중...")
def load_stock_listing():
    """KRX 전체 종목 리스트 로딩"""
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq], ignore_index=True)
        
        if 'Symbol' in stocks.columns:
            stocks = stocks.rename(columns={'Symbol': 'Code'})
        elif 'code' in stocks.columns:
            stocks = stocks.rename(columns={'code': 'Code'})
        
        stocks['Code'] = stocks['Code'].astype(str).str.zfill(6)
        result = stocks[['Code', 'Name']].dropna().drop_duplicates()
        return result
    except Exception:
        return pd.DataFrame(columns=['Code', 'Name'])

# -----------------------------
# 종목명/코드 변환
# -----------------------------
def resolve_stock_input(user_input, stock_list):
    """사용자 입력을 종목코드로 변환 (유사도 매칭 포함)"""
    if stock_list.empty:
        return None, None, None
    
    user_input = user_input.strip()
    
    if user_input.isdigit():
        code = user_input.zfill(6)
        match = stock_list[stock_list['Code'] == code]
        if len(match) > 0:
            return code, match.iloc[0]['Name'], 'exact_code'
        else:
            return None, None, None
    
    exact_match = stock_list[stock_list['Name'] == user_input]
    if len(exact_match) == 1:
        return exact_match.iloc[0]['Code'], exact_match.iloc[0]['Name'], 'exact_name'
    elif len(exact_match) > 1:
        return exact_match.iloc[0]['Code'], exact_match.iloc[0]['Name'], 'exact_name'
    
    partial_match = stock_list[stock_list['Name'].str.contains(user_input, case=False, na=False)]
    if len(partial_match) == 1:
        return partial_match.iloc[0]['Code'], partial_match.iloc[0]['Name'], 'exact_name'
    elif len(partial_match) > 1:
        partial_match = partial_match.copy()
        partial_match['name_len'] = partial_match['Name'].str.len()
        partial_match = partial_match.sort_values('name_len')
        return partial_match.iloc[0]['Code'], partial_match.iloc[0]['Name'], 'exact_name'
    
    all_names = stock_list['Name'].tolist()
    close_matches = get_close_matches(user_input, all_names, n=3, cutoff=0.6)
    
    if close_matches:
        best_match = close_matches[0]
        match_row = stock_list[stock_list['Name'] == best_match].iloc[0]
        return match_row['Code'], match_row['Name'], 'fuzzy_name'
    
    return None, None, None

# -----------------------------
# 데이터 로딩
# -----------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_data(code, days=260):
    """순수 함수: 데이터만 반환"""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = fdr.DataReader(code, start, end)
        if df is None or len(df) < 120:
            return None
        return df
    except Exception:
        return None

def add_indicators(df):
    """기술 지표 계산"""
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolAvg60"] = df["Volume"].rolling(60).mean()

    # Bollinger(60)
    win = 60
    mid = df["Close"].rolling(win).mean()
    std = df["Close"].rolling(win).std(ddof=0)
    df["BB_MID60"] = mid
    df["BB_UP60"] = mid + 2 * std
    df["BB_DN60"] = mid - 2 * std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df["BBW60"] = np.where(
            (df["BB_MID60"] > 0) & df["BB_MID60"].notna(),
            (df["BB_UP60"] - df["BB_DN60"]) / df["BB_MID60"],
            np.nan
        )
    
    return df

# -----------------------------
# 스윙 저점 탐지
# -----------------------------
def last_swing_low(df, left=2, right=2, lookback=60):
    """가장 최근 스윙저점 반환"""
    x = df.tail(lookback).copy()
    if len(x) < left + right + 1:
        return None
    
    lows = x["Low"].values
    idxs = x.index.to_list()

    pivots = []
    for i in range(left, len(x) - right):
        window = lows[i-left:i+right+1]
        if lows[i] == np.min(window):
            pivots.append((idxs[i], float(lows[i])))

    if not pivots:
        return None
    return pivots[-1]

# -----------------------------
# 손절 후보
# -----------------------------
def stop_candidates(df):
    """손절 후보 2개 반환"""
    last10_low = float(df["Low"].tail(10).min())
    swing = last_swing_low(df, left=2, right=2, lookback=60)
    swing_low = float(swing[1]) if swing else None
    swing_date = swing[0] if swing else None
    return last10_low, swing_low, swing_date

# -----------------------------
# 평가 함수
# -----------------------------
def risk_ok(entry, stop, max_loss_pct=8.0):
    """손절폭 체크"""
    if entry <= 0:
        return False, 0.0
    loss_pct = (stop - entry) / entry * 100.0
    return (loss_pct >= -max_loss_pct), loss_pct

def volume_surge_ratio(df, which="VolAvg20"):
    """거래량 비율"""
    row = df.iloc[-1]
    base = row[which]
    if pd.isna(base) or base <= 0:
        return None
    if pd.isna(row["Volume"]):
        return None
    return float(row["Volume"] / base)

def bbw_percentile(df, lookback=252):
    """밴드폭 백분위수"""
    x = df["BBW60"].dropna().tail(lookback)
    if len(x) < 60:
        return None, None
    today = float(x.iloc[-1])
    pct = float((x.rank(pct=True).iloc[-1]) * 100)
    return today, pct

# -----------------------------
# 체크리스트 함수들
# -----------------------------
def check_bb_breakout(df, entry):
    """60일 볼린저 상단 돌파"""
    results = []
    close = df.iloc[-1]["Close"]
    bb_up = df.iloc[-1]["BB_UP60"]
    
    if pd.isna(bb_up):
        results.append((False, "BB_UP60=N/A"))
    else:
        passed = bool(close > bb_up)
        results.append((passed, f"Close={close:.0f}, BB_UP={bb_up:.0f}"))
    
    vol_ratio = volume_surge_ratio(df, "VolAvg20")
    if vol_ratio is None:
        results.append((False, "Vol/Avg20=N/A"))
    else:
        passed = vol_ratio >= 1.5
        results.append((passed, f"Vol/Avg20={vol_ratio:.2f}"))
    
    bbw_val, bbw_pct = bbw_percentile(df)
    if bbw_pct is None:
        results.append((False, "BBW_pct=N/A"))
    else:
        passed = bbw_pct <= 30.0
        results.append((passed, f"BBW_pct={bbw_pct:.1f}%"))
    
    if pd.isna(bb_up):
        results.append((False, "BB_UP60=N/A"))
    else:
        passed = entry <= bb_up * 1.03
        results.append((passed, f"Entry={entry:.0f}, BB_UP*1.03={bb_up*1.03:.0f}"))
    
    return results

def check_ma20_breakout(df, entry):
    """20일선 돌파/리클레임"""
    results = []
    close = df.iloc[-1]["Close"]
    ma20 = df.iloc[-1]["MA20"]
    
    if pd.isna(ma20):
        results.append((False, "MA20=N/A"))
    else:
        passed = bool(close > ma20)
        results.append((passed, f"Close={close:.0f}, MA20={ma20:.0f}"))
    
    vol_ratio = volume_surge_ratio(df, "VolAvg20")
    if vol_ratio is None:
        results.append((False, "Vol/Avg20=N/A"))
    else:
        passed = vol_ratio >= 1.5
        results.append((passed, f"Vol/Avg20={vol_ratio:.2f}"))
    
    ma20_series = df["MA20"].dropna()
    if len(ma20_series) < 6:
        results.append((False, "MA20 데이터 부족"))
    else:
        delta = float(ma20_series.iloc[-1] - ma20_series.iloc[-6])
        passed = delta >= 0
        results.append((passed, f"ΔMA20(5d)={delta:.2f}"))
    
    if pd.isna(ma20):
        results.append((False, "MA20=N/A"))
    else:
        passed = entry <= ma20 * 1.05
        results.append((passed, f"Entry={entry:.0f}, MA20*1.05={ma20*1.05:.0f}"))
    
    return results

def check_trend_pullback(df, entry, ma_type="MA20"):
    """추세추종 눌림 (MA20 또는 MA50)"""
    results = []
    close = df.iloc[-1]["Close"]
    ma = df.iloc[-1][ma_type]
    
    # A1. 기준선 위
    if pd.isna(ma):
        results.append((False, f"{ma_type}=N/A"))
    else:
        passed = bool(close > ma)
        results.append((passed, f"Close={close:.0f}, {ma_type}={ma:.0f}"))
    
    # A2. 기준선 상승
    ma_series = df[ma_type].dropna()
    if len(ma_series) < 6:
        results.append((False, f"{ma_type} 데이터 부족"))
    else:
        delta = float(ma_series.iloc[-1] - ma_series.iloc[-6])
        passed = delta >= 0
        results.append((passed, f"Δ{ma_type}(5d)={delta:.2f}"))
    
    # B1. 근접/터치 (±2%)
    recent = df.tail(10)
    if pd.isna(ma):
        results.append((False, f"{ma_type}=N/A"))
    else:
        tolerance = 0.02
        touch = any((recent["Low"] >= ma * (1 - tolerance)) & (recent["Low"] <= ma * (1 + tolerance)))
        low_min = float(recent["Low"].min())
        results.append((touch, f"최근10일 저가={low_min:.0f}, {ma_type}±2%=[{ma*(1-tolerance):.0f}, {ma*(1+tolerance):.0f}]"))
    
    # B3. 거래량 감소 (눌림 구간)
    pullback_vol = recent["Volume"].tail(3).mean()
    avg_vol = df["VolAvg20"].iloc[-1]
    if pd.isna(avg_vol) or avg_vol <= 0:
        results.append((False, "VolAvg20=N/A"))
    else:
        passed = pullback_vol < avg_vol
        results.append((passed, f"눌림Vol(3d avg)={pullback_vol:.0f}, VolAvg20={avg_vol:.0f}"))
    
    # C1. 턴업 확인 (종가 > 전일 고가)
    if len(df) < 2:
        results.append((False, "데이터 부족"))
    else:
        prev_high = df["High"].iloc[-2]
        passed = close > prev_high
        results.append((passed, f"Close={close:.0f}, 전일High={prev_high:.0f}"))
    
    return results

# -----------------------------
# 셋업 정의
# -----------------------------
SETUPS = {
    "60일 볼린저 상단 돌파": {
        "fn": check_bb_breakout,
        "labels": [
            "종가가 60일 볼린저 상단 위에서 마감",
            "거래량 급증(당일 >= 20일 평균의 1.5배)",
            "밴드 수축(60일 밴드폭이 최근 1년 중 하위 30%)",
            "진입가가 과도하게 확장되지 않음(상단밴드 대비 +3% 이내)"
        ],
        "weights": [25, 35, 20, 20],
        "ma_select": None
    },
    "20일선 돌파/리클레임": {
        "fn": check_ma20_breakout,
        "labels": [
            "종가가 20일선 위에서 마감",
            "거래량 급증(당일 >= 20일 평균의 1.5배)",
            "20일선 기울기(최근 5일 MA20 상승 또는 평탄)",
            "진입가가 20일선에서 너무 멀지 않음(+5% 이내)"
        ],
        "weights": [25, 35, 20, 20],
        "ma_select": None
    },
    "추세추종 눌림(MA20/MA50)": {
        "fn": check_trend_pullback,
        "labels": [
            "종가가 기준선 위",
            "기준선 기울기 상승(최근 5일)",
            "최근 10일 내 저가가 기준선 ±2% 터치",
            "눌림 구간 거래량 감소(3일 평균 < VolAvg20)",
            "턴업 확인(종가 > 전일 고가)"
        ],
        "weights": [20, 15, 25, 20, 20],
        "ma_select": ["MA20", "MA50"]
    }
}

# -----------------------------
# 종목 리스트 로딩
# -----------------------------
stock_listing = load_stock_listing()

# -----------------------------
# UI 입력
# -----------------------------
st.markdown("### 📥 입력")
colA, colB, colC = st.columns([2.0, 2.0, 2.0])

with colA:
    user_stock_input = st.text_input(
        "종목코드 또는 종목명", 
        value="", 
        placeholder="예: 005930 또는 삼성전자",
        help="종목코드(6자리) 또는 종목명 입력 (오타 자동 보정)"
    )

with colB:
    setup_name = st.selectbox("상황(셋업) 선택", list(SETUPS.keys()))

with colC:
    entry_price = st.number_input("내 진입가(원) (기본=현재가)", min_value=0.0, value=0.0, step=100.0)

# 추세추종 눌림 셋업일 때만 MA 선택
selected_ma = None
if SETUPS[setup_name]["ma_select"] is not None:
    selected_ma = st.radio("기준 이동평균선 선택", SETUPS[setup_name]["ma_select"], horizontal=True)

st.divider()

# -----------------------------
# 종목 입력 검증
# -----------------------------
if not user_stock_input.strip():
    st.info("👆 종목코드(예: 005930) 또는 종목명(예: 삼성전자)을 입력하세요.")
    st.stop()

code, stock_name, match_type = resolve_stock_input(user_stock_input, stock_listing)

if code is None:
    st.error(f"❌ 종목을 찾을 수 없습니다: '{user_stock_input}'")
    st.warning("💡 힌트: 종목코드 6자리(예: 005930) 또는 정확한 종목명(예: 삼성전자)을 입력하세요.")
    st.stop()

if match_type == 'exact_code':
    st.success(f"✅ 종목 확인: **{stock_name}** ({code})")
elif match_type == 'exact_name':
    st.success(f"✅ 종목 확인: **{stock_name}** ({code})")
elif match_type == 'fuzzy_name':
    st.warning(f"🔍 '{user_stock_input}' → **{stock_name}** ({code})로 자동 보정되었습니다.")

# -----------------------------
# 데이터 로딩
# -----------------------------
df = load_data(code)

if df is None:
    st.error(f"❌ 데이터 로딩 실패: {stock_name}({code})의 데이터가 부족하거나 없습니다(최소 120일 필요).")
    st.warning("💡 힌트: 상장폐지 종목이거나 데이터가 충분하지 않을 수 있습니다.")
    st.stop()

data_source_info = f"📊 **데이터 소스**: FinanceDataReader (크롤링 기반) | **최종 업데이트**: {df.index[-1].strftime('%Y-%m-%d')}"
st.caption(data_source_info)

df = add_indicators(df)
last = df.iloc[-1]
current_price = float(last["Close"])

if entry_price == 0.0:
    entry_price = current_price

# -----------------------------
# 손절 후보 & 8% 필터
# -----------------------------
low10, swing_low, swing_date = stop_candidates(df)

cands = []
ok1, loss1 = risk_ok(entry_price, low10, max_loss_pct=8.0)
cands.append({
    "손절 후보": "최근 10일 최저가",
    "손절가": low10,
    "손절폭(%)": loss1,
    "유효(<=8%)": ok1,
    "근거": "Low(min 10d)"
})

if swing_low is not None:
    ok2, loss2 = risk_ok(entry_price, swing_low, max_loss_pct=8.0)
    cands.append({
        "손절 후보": f"스윙 저점({str(swing_date)[:10]})",
        "손절가": swing_low,
        "손절폭(%)": loss2,
        "유효(<=8%)": ok2,
        "근거": "Pivot low"
    })

cand_df = pd.DataFrame(cands)
valid_cands = cand_df[cand_df["유효(<=8%)"] == True].copy()

# -----------------------------
# 결과 표시
# -----------------------------
st.markdown("---")
st.subheader(f"📌 {stock_name} ({code}) | 종가(최근 일봉) 기준 평가")
m1, m2, m3, m4 = st.columns(4)
m1.metric("현재가(종가)", f"{current_price:,.0f}원")
m2.metric("내 진입가", f"{entry_price:,.0f}원")
m3.metric("선택 셋업", setup_name)
m4.metric("데이터 모드", "종가 확정(EOD)")

st.markdown("### 🛑 손절 후보(C) & 8% 룰")
show = cand_df.copy()
show["손절가"] = show["손절가"].map(lambda x: f"{x:,.0f}원")
show["손절폭(%)"] = show["손절폭(%)"].map(lambda x: f"{x:.2f}%")
st.dataframe(show, use_container_width=True, hide_index=True)

if len(valid_cands) == 0:
    st.error("❌ **FAIL**: 손절 후보(10일저점/스윙저점) 모두 진입가 대비 -8%를 초과합니다.")
    st.warning("💡 **오닐식 8% 룰 위반**: 손실 리스크가 너무 커서 진입이 부적합합니다. 진입가를 낮추거나 다른 종목을 고려하세요.")
    st.stop()

valid_cands = valid_cands.sort_values("손절폭(%)", ascending=False)
chosen = valid_cands.iloc[0]
chosen_stop = float(chosen["손절가"])
chosen_loss = float(chosen["손절폭(%)"])

st.success(f"✅ **유효 손절 선택**: {chosen['손절 후보']} | 손절가 **{chosen_stop:,.0f}원** | 손절폭 **{chosen_loss:.2f}%**")

# -----------------------------
# 체크리스트 평가
# -----------------------------
setup = SETUPS[setup_name]

# 추세추종 눌림의 경우 MA 타입 전달
if selected_ma:
    check_results = setup["fn"](df, entry_price, selected_ma)
else:
    check_results = setup["fn"](df, entry_price)

labels = setup["labels"]
weights = setup["weights"]

rows = []
score = 0
max_score = sum(weights)

for i, (passed, detail) in enumerate(check_results):
    w = weights[i]
    s = w if passed else 0
    score += s
    rows.append({
        "항목": labels[i],
        "우선순위(가중치)": w,
        "통과": "✅" if passed else "❌",
        "근거(계산값)": detail,
    })

final_score = int(round((score / max_score) * 100))

st.markdown("### ✅ 체크리스트 (상황별)")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### 🎯 신뢰도")
s1, s2, s3 = st.columns(3)
s1.metric("신뢰도(0~100)", f"{final_score}점")
s2.metric("선택 손절가", f"{chosen_stop:,.0f}원")
s3.metric("손절폭(%)", f"{chosen_loss:.2f}%")

if final_score >= 80:
    st.success("🎯 **신뢰도 높음**: 체크리스트 대부분 통과 (진입 고려 가능)")
elif final_score >= 60:
    st.info("⚠️ **신뢰도 보통**: 일부 항목 미달 (추가 확인 필요)")
else:
    st.warning("⚡ **신뢰도 낮음**: 주요 항목 미통과 (진입 재고려 권장)")

# -----------------------------
# 차트
# -----------------------------
st.markdown("### 📈 차트(120일) - 진입/손절 표시")
chart_df = df.tail(120)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["Open"],
    high=chart_df["High"],
    low=chart_df["Low"],
    close=chart_df["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA20"], name="MA20", 
                         line=dict(color="blue", width=1.5)))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA50"], name="MA50", 
                         line=dict(color="purple", width=1.5)))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_UP60"], name="BB_UP60", 
                         line=dict(color="gray", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_MID60"], name="BB_MID60", 
                         line=dict(color="gray", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_DN60"], name="BB_DN60", 
                         line=dict(color="gray", width=1, dash="dot")))

fig.add_trace(go.Scatter(
    x=[chart_df.index[0], chart_df.index[-1]],
    y=[entry_price, entry_price],
    name=f"Entry ({entry_price:,.0f}원)",
    line=dict(color="green", width=2.5, dash="dash"),
    mode="lines"
))
fig.add_trace(go.Scatter(
    x=[chart_df.index[0], chart_df.index[-1]],
    y=[chosen_stop, chosen_stop],
    name=f"Stop ({chosen_stop:,.0f}원)",
    line=dict(color="red", width=2.5, dash="dash"),
    mode="lines"
))

title_text = f"{stock_name}({code}) | {setup_name}"
if selected_ma:
    title_text += f" ({selected_ma})"
title_text += f" | Entry: {entry_price:,.0f}원 | Stop: {chosen_stop:,.0f}원 | 신뢰도: {final_score}점"

fig.update_layout(
    height=600,
    title=title_text,
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 하단 경고문
# -----------------------------
st.divider()
st.warning("""
⚠️ **중요 공지**:
- 본 평가는 **일봉 종가 기준(EOD)** 규칙 평가이며, 실시간(장중) 확정 신호가 아닙니다.
- 데이터는 FinanceDataReader(크롤링 기반)로 수집되며, 소스에 따라 거래량/가격 차이가 있을 수 있습니다.
- **중요한 매매 결정 전 반드시 증권사 HTS/MTS로 재확인하세요.**
- 본 앱은 참고용이며, 투자 손실에 대한 책임은 사용자에게 있습니다.
""")

st.caption(f"💾 마지막 데이터 업데이트: {df.index[-1].strftime('%Y-%m-%d')} | 평가 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


