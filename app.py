import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# -----------------------------------------------------------
# 두산로보틱스 진단 설정 (엄격 모드)
# -----------------------------------------------------------
st.set_page_config(page_title="두산로보틱스 진단", layout="wide")
st.title("🕵️‍♂️ 두산로보틱스 정밀 진단 (최초 엄격 기준)")

# 1. 데이터 가져오기
code = '454910' # 두산로보틱스
name = "두산로보틱스"
df = fdr.DataReader(code, '2024-01-01')

if df is None or len(df) == 0:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

current_price = df['Close'].iloc[-1]
st.metric(f"{name} ({code})", f"{current_price:,.0f}원")

# -----------------------------------------------------------
# 1. Stage 2 (추세) 진단
# -----------------------------------------------------------
st.header("1. Stage 2 (추세) 진단")

ma50 = df['Close'].rolling(50).mean().iloc[-1]
ma150 = df['Close'].rolling(150).mean().iloc[-1]
ma200 = df['Close'].rolling(200).mean().iloc[-1]
close = df['Close'].iloc[-1]

# 200일선 데이터 확보 확인
if len(df) < 200:
    st.warning(f"⚠️ 상장일이 짧아 데이터 부족 ({len(df)}일). 200일선 계산 불가할 수 있음.")
    ma200_val = ma200 if not np.isnan(ma200) else 0
else:
    ma200_val = ma200

check_list = [
    ("정배열 상태인가?", close > ma50 > ma150 > ma200_val, f"현재: {close:,.0f} > 50일: {ma50:,.0f} > 150일: {ma150:,.0f} > 200일: {ma200_val:,.0f}"),
    ("주가가 200일선 위에 있나?", close > ma200_val, f"이격도: {(close/ma200_val - 1)*100:.1f}% 위"),
]

for title, result, desc in check_list:
    st.write(f"{'✅' if result else '❌'} **{title}**: {desc}")

st.divider()

# -----------------------------------------------------------
# 2. VCP 패턴 진단 (여기가 핵심!)
# -----------------------------------------------------------
st.header("2. VCP 패턴 진단 (엄격 기준)")

# 최초 코드의 설정: distance=5 (돋보기 모드)
recent = df.tail(120).copy()
recent['atr'] = (recent['High'] - recent['Low']) / recent['Close']
peaks, _ = find_peaks(recent['High'].values, distance=5) 

if len(peaks) < 2:
    st.error(f"❌ 파동 개수 부족: {len(peaks)}개 (최소 3개 필요했던 기준)")
else:
    st.success(f"✅ 파동 개수: {len(peaks)}개 (Distance=5 기준)")
    
    # 변동성 계산
    volatilities = []
    for i in range(len(peaks)-1):
        vol = recent['atr'].iloc[peaks[i]:peaks[i+1]].mean()
        volatilities.append(vol)
    
    # 마지막 파동
    last_peak_idx = peaks[-1]
    last_vol = recent['atr'].iloc[last_peak_idx:].mean()
    volatilities.append(last_vol)
    
    # 최근 3개만 비교
    check_vols = volatilities[-3:] if len(volatilities) >= 3 else volatilities
    
    st.write("---")
    st.subheader("📊 파동별 변동성 (엄격 기준)")
    
    cols = st.columns(len(check_vols))
    for i, v in enumerate(check_vols):
        cols[i].metric(f"파동 {i+1}", f"{v:.2%}")

    # [범인 후보 1] 수축 여부 (과거보다 줄었나?)
    # 최초 코드는 '순차적 감소' 또는 '마지막 < 첫번째'를 엄격히 봤음
    cond_shrink = check_vols[-1] < check_vols[0]
    st.write(f"{'✅' if cond_shrink else '❌'} **변동성이 줄어들었는가?** (마지막 {check_vols[-1]:.2%} < 첫번째 {check_vols[0]:.2%})")
    
    # [범인 후보 2] 마지막 파동 크기 (4% 이내인가?)
    cond_tight = last_vol <= 0.04
    st.write(f"{'✅' if cond_tight else '❌'} **마지막 파동이 4% 이내인가?** (현재: {last_vol:.2%})")
    if not cond_tight:
        st.caption("👉 최초 코드는 4% 넘으면 '변동성이 너무 크다'고 탈락시켰습니다.")

st.divider()

# -----------------------------------------------------------
# 3. 거래량 & Pivot 진단
# -----------------------------------------------------------
st.header("3. 거래량 & Pivot 진단")

vol_ma50 = df['Volume'].rolling(50).mean().iloc[-1]
last_wave_vol = recent['Volume'].iloc[last_peak_idx:].mean()

# [범인 후보 3] 거래량 Dry-up (평균 이하인가?)
vol_ratio = last_wave_vol / vol_ma50
cond_vol = vol_ratio <= 1.0 # 최초 코드는 1.0배 (평균 이하)를 원했음
st.write(f"{'✅' if cond_vol else '❌'} **거래량이 50일 평균 이하인가?**")
st.caption(f"내 수치: {vol_ratio:.2f}배 (1.0배 넘으면 탈락)")

# [범인 후보 4] Pivot 거리 (8% 이내인가?)
pivot = recent['High'].iloc[last_peak_idx]
pivot_dist = (pivot - close) / close

cond_pivot = 0 <= pivot_dist <= 0.08
st.write(f"{'✅' if cond_pivot else '❌'} **Pivot 거리가 0~8% 이내인가?**")
st.caption(f"Pivot: {pivot:,.0f}원 / 현재가: {close:,.0f}원 / 거리: {pivot_dist*100:.1f}%")

if pivot_dist < 0:
    st.warning("👉 이미 Pivot을 돌파해버려서(-값) 탈락했을 수 있습니다.")
elif pivot_dist > 0.08:
    st.warning("👉 Pivot과 너무 멀어서(8% 초과) 탈락했습니다.")
