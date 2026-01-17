import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
# 데이터 로딩 (캐시 함수: 순수 함수로 유지)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_data(code, days=260):
    """
    순수 함수: 데이터 로딩만 수행, UI 출력 없음
    성공 시 DataFrame 반환, 실패 시 None 반환
    """
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
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolAvg60"] = df["Volume"].rolling(60).mean()

    # Bollinger(60)
    win = 60
    mid = df["Close"].rolling(win).mean()
    std = df["Close"].rolling(win).std(ddof=0)
    df["BB_MID60"] = mid
    df["BB_UP60"] = mid + 2 * std
    df["BB_DN60"] = mid - 2 * std
    
    # BBW60: NaN/0 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        df["BBW60"] = np.where(
            (df["BB_MID60"] > 0) & df["BB_MID60"].notna(),
            (df["BB_UP60"] - df["BB_DN60"]) / df["BB_MID60"],
            np.nan
        )
    
    return df

# -----------------------------
# 스윙 저점(단순) 탐지
# -----------------------------
def last_swing_low(df, left=2, right=2, lookback=60):
    """
    가장 최근 스윙저점(피벗 로우) 하나를 반환.
    정의: i일의 Low가 [i-left .. i+right] 중 최저이면 스윙저점.
    """
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
    return pivots[-1]  # 가장 최근

# -----------------------------
# 손절 후보(C): 10일 최저가 & 스윙 저점
# -----------------------------
def stop_candidates(df):
    """손절 후보 2개 반환"""
    last10_low = float(df["Low"].tail(10).min())
    swing = last_swing_low(df, left=2, right=2, lookback=60)
    swing_low = float(swing[1]) if swing else None
    swing_date = swing[0] if swing else None
    return last10_low, swing_low, swing_date

# -----------------------------
# 체크리스트 평가 프레임
# -----------------------------
def risk_ok(entry, stop, max_loss_pct=8.0):
    """손절폭이 -max_loss_pct 이내인지 체크"""
    if entry <= 0:
        return False, 0.0
    loss_pct = (stop - entry) / entry * 100.0
    return (loss_pct >= -max_loss_pct), loss_pct

def volume_surge_ratio(df, which="VolAvg20"):
    """거래량 비율 계산"""
    row = df.iloc[-1]
    base = row[which]
    if pd.isna(base) or base <= 0:
        return None
    if pd.isna(row["Volume"]):
        return None
    return float(row["Volume"] / base)

def bbw_percentile(df, lookback=252):
    """밴드폭 백분위수 계산"""
    x = df["BBW60"].dropna().tail(lookback)
    if len(x) < 60:
        return None, None
    today = float(x.iloc[-1])
    pct = float((x.rank(pct=True).iloc[-1]) * 100)
    return today, pct

# -----------------------------
# 셋업별 체크리스트 (중복 호출 제거)
# -----------------------------
def check_bb_breakout(df, entry):
    """60일 볼린저 상단 돌파형 체크리스트"""
    results = []
    
    # 1. 종가가 BB 상단 위
    close = df.iloc[-1]["Close"]
    bb_up = df.iloc[-1]["BB_UP60"]
    if pd.isna(bb_up):
        results.append((False, "BB_UP60=N/A"))
    else:
        passed = bool(close > bb_up)
        results.append((passed, f"Close={close:.0f}, BB_UP={bb_up:.0f}"))
    
    # 2. 거래량 급증
    vol_ratio = volume_surge_ratio(df, "VolAvg20")
    if vol_ratio is None:
        results.append((False, "Vol/Avg20=N/A"))
    else:
        passed = vol_ratio >= 1.5
        results.append((passed, f"Vol/Avg20={vol_ratio:.2f}"))
    
    # 3. 밴드 수축
    bbw_val, bbw_pct = bbw_percentile(df)
    if bbw_pct is None:
        results.append((False, "BBW_pct=N/A"))
    else:
        passed = bbw_pct <= 30.0
        results.append((passed, f"BBW_pct={bbw_pct:.1f}%"))
    
    # 4. 확장 과다 방지
    if pd.isna(bb_up):
        results.append((False, "BB_UP60=N/A"))
    else:
        passed = entry <= bb_up * 1.03
        results.append((passed, f"Entry={entry:.0f}, BB_UP*1.03={bb_up*1.03:.0f}"))
    
    return results

def check_ma20_breakout(df, entry):
    """20일선 돌파형 체크리스트"""
    results = []
    
    # 1. 종가가 MA20 위
    close = df.iloc[-1]["Close"]
    ma20 = df.iloc[-1]["MA20"]
    if pd.isna(ma20):
        results.append((False, "MA20=N/A"))
    else:
        passed = bool(close > ma20)
        results.append((passed, f"Close={close:.0f}, MA20={ma20:.0f}"))
    
    # 2. 거래량 급증
    vol_ratio = volume_surge_ratio(df, "VolAvg20")
    if vol_ratio is None:
        results.append((False, "Vol/Avg20=N/A"))
    else:
        passed = vol_ratio >= 1.5
        results.append((passed, f"Vol/Avg20={vol_ratio:.2f}"))
    
    # 3. MA20 기울기
    ma20_series = df["MA20"].dropna()
    if len(ma20_series) < 6:
        results.append((False, "MA20 데이터 부족"))
    else:
        delta = float(ma20_series.iloc[-1] - ma20_series.iloc[-6])
        passed = delta >= 0
        results.append((passed, f"ΔMA20(5d)={delta:.2f}"))
    
    # 4. MA20 거리
    if pd.isna(ma20):
        results.append((False, "MA20=N/A"))
    else:
        passed = entry <= ma20 * 1.05
        results.append((passed, f"Entry={entry:.0f}, MA20*1.05={ma20*1.05:.0f}"))
    
    return results

SETUPS = {
    "60일 볼린저 상단 돌파": {
        "fn": check_bb_breakout,
        "labels": [
            "종가가 60일 볼린저 상단 위에서 마감",
            "거래량 급증(당일 >= 20일 평균의 1.5배)",
            "밴드 수축(60일 밴드폭이 최근 1년 중 하위 30%)",
            "진입가가 과도하게 확장되지 않음(상단밴드 대비 +3% 이내)"
        ],
        "weights": [25, 35, 20, 20]
    },
    "20일선 돌파/리클레임": {
        "fn": check_ma20_breakout,
        "labels": [
            "종가가 20일선 위에서 마감",
            "거래량 급증(당일 >= 20일 평균의 1.5배)",
            "20일선 기울기(최근 5일 MA20 상승 또는 평탄)",
            "진입가가 20일선에서 너무 멀지 않음(+5% 이내)"
        ],
        "weights": [25, 35, 20, 20]
    }
}

# -----------------------------
# UI 입력
# -----------------------------
st.markdown("### 📥 입력")
colA, colB, colC = st.columns([2.0, 2.0, 2.0])

with colA:
    code = st.text_input("종목코드(예: 005930)", value="", placeholder="005930")

with colB:
    setup_name = st.selectbox("상황(셋업) 선택", list(SETUPS.keys()))

with colC:
    entry_price = st.number_input("내 진입가(원) (기본=현재가)", min_value=0.0, value=0.0, step=100.0)

st.divider()

# -----------------------------
# 입력 검증
# -----------------------------
if not code.strip():
    st.info("👆 종목코드를 입력하세요.")
    st.stop()

# -----------------------------
# 데이터 로딩 (캐시 함수 외부에서 UI 처리)
# -----------------------------
df = load_data(code.strip())

if df is None:
    st.error("❌ 데이터 로딩 실패: 종목코드가 잘못되었거나 데이터가 부족합니다(최소 120일 필요).")
    st.warning("💡 힌트: 종목코드 6자리를 정확히 입력했는지 확인하세요(예: 005930).")
    st.stop()

# 데이터 소스 투명성 표시
data_source_info = f"📊 **데이터 소스**: FinanceDataReader (크롤링 기반) | **최종 업데이트**: {df.index[-1].strftime('%Y-%m-%d')}"
st.caption(data_source_info)

df = add_indicators(df)
last = df.iloc[-1]
current_price = float(last["Close"])

if entry_price == 0.0:
    entry_price = current_price

# -----------------------------
# 손절 후보(C) 생성 & 8% 필터
# -----------------------------
low10, swing_low, swing_date = stop_candidates(df)

cands = []
# 후보 1: 10일 최저가
ok1, loss1 = risk_ok(entry_price, low10, max_loss_pct=8.0)
cands.append({
    "손절 후보": "최근 10일 최저가",
    "손절가": low10,
    "손절폭(%)": loss1,
    "유효(<=8%)": ok1,
    "근거": "Low(min 10d)"
})

# 후보 2: 최근 스윙저점
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
st.subheader(f"📌 {code} | 종가(최근 일봉) 기준 평가")
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

# 유효 후보 중 가장 타이트한(절댓값이 작은) 손절 선택
valid_cands = valid_cands.sort_values("손절폭(%)", ascending=False)
chosen = valid_cands.iloc[0]
chosen_stop = float(chosen["손절가"])
chosen_loss = float(chosen["손절폭(%)"])

st.success(f"✅ **유효 손절 선택**: {chosen['손절 후보']} | 손절가 **{chosen_stop:,.0f}원** | 손절폭 **{chosen_loss:.2f}%**")

# -----------------------------
# 체크리스트 평가 + 점수
# -----------------------------
setup = SETUPS[setup_name]
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

# 신뢰도 해석
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

# 지표들
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA20"], name="MA20", 
                         line=dict(color="blue", width=1.5)))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_UP60"], name="BB_UP60", 
                         line=dict(color="gray", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_MID60"], name="BB_MID60", 
                         line=dict(color="gray", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_DN60"], name="BB_DN60", 
                         line=dict(color="gray", width=1, dash="dot")))

# 진입/손절(수평선)
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

fig.update_layout(
    height=600,
    title=f"{code} | {setup_name} | Entry: {entry_price:,.0f}원 | Stop: {chosen_stop:,.0f}원 | 신뢰도: {final_score}점",
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 하단 경고문 (데이터 품질 투명성)
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
