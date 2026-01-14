import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

st.title("🕵️‍♂️ 가온전선 정밀 진단 키트")

# 1. 가온전선 데이터 가져오기
code = '001440' # 가온전선
df = fdr.DataReader(code, '2024-01-01')

st.header(f"가온전선 ({code}) 진단 결과")
st.metric("현재가", f"{df['Close'].iloc[-1]:,.0f}원")

# -----------------------------------------------------------
# 진단 로직 (사용자님 앱과 동일한 기준)
# -----------------------------------------------------------

# 1. Stage 2 진단
st.subheader("1. Stage 2 (추세) 진단")
ma50 = df['Close'].rolling(50).mean().iloc[-1]
ma150 = df['Close'].rolling(150).mean().iloc[-1]
ma200 = df['Close'].rolling(200).mean().iloc[-1]
close = df['Close'].iloc[-1]
low_52 = df['Low'].tail(252).min()
high_52 = df['High'].tail(252).max()

check_list = [
    ("200일선 위에 있는가?", close >= ma200, f"현재: {close} / 200일선: {ma200:.0f}"),
    ("200일선 상승 중인가?", df['Close'].rolling(200).mean().iloc[-1] > df['Close'].rolling(200).mean().iloc[-22], "1개월 전 대비 상승"),
    ("바닥 대비 25% 상승했나?", close >= low_52 * 1.25, f"현재: {close} / 바닥+25%: {low_52*1.25:.0f}"),
    ("고점 대비 -30% 이내인가?", close >= high_52 * 0.70, f"현재: {close} / 고점-30%: {high_52*0.70:.0f}")
]

for title, result, desc in check_list:
    st.write(f"{'✅' if result else '❌'} **{title}**: {desc}")

st.divider()

# 2. VCP 패턴 진단
st.subheader("2. VCP 패턴 (모양) 진단")
recent = df.tail(120).copy()
recent['atr'] = (recent['High'] - recent['Low']) / recent['Close']
peaks, _ = find_peaks(recent['High'].values, distance=5)

if len(peaks) < 2:
    st.error(f"❌ 파동 개수 부족: {len(peaks)}개 (최소 2개 필요)")
else:
    st.success(f"✅ 파동 개수 충족: {len(peaks)}개")
    
    # 변동성 계산
    volatilities = []
    for i in range(len(peaks)-1):
        vol = recent['atr'].iloc[peaks[i]:peaks[i+1]].mean()
        volatilities.append(vol)
    volatilities.append(recent['atr'].iloc[peaks[-1]:].mean())
    
    check_vols = volatilities[-3:] if len(volatilities) >= 3 else volatilities
    
    st.write("---")
    st.write("📊 **파동별 변동성 (수치)**")
    for i, v in enumerate(check_vols):
        st.text(f"파동 {i+1}: {v:.2%} (낮을수록 좋음)")

    # 1. 수축 여부
    cond_shrink = check_vols[-1] <= check_vols[0]
    st.write(f"{'✅' if cond_shrink else '❌'} **변동성이 줄어들었는가?** (마지막 {check_vols[-1]:.2%} vs 첫번째 {check_vols[0]:.2%})")
    
    # 2. 마지막 파동 크기
    cond_tight = check_vols[-1] <= 0.10
    st.write(f"{'✅' if cond_tight else '❌'} **마지막 파동이 10% 이내인가?** (현재: {check_vols[-1]:.2%})")

st.divider()

# 3. 거래량 & Pivot 진단 (가장 유력한 범인)
st.subheader("3. 거래량 & Pivot 진단")

vol_ma50 = df['Volume'].rolling(50).mean().iloc[-1]
last_peak_idx = peaks[-1]
last_wave_vol = recent['Volume'].iloc[last_peak_idx:].mean()

# 거래량 비율
vol_ratio = last_wave_vol / vol_ma50
cond_vol = vol_ratio <= 1.2
st.write(f"{'✅' if cond_vol else '❌'} **거래량이 평균 1.2배 이하인가?**")
st.caption(f"내 수치: {vol_ratio:.2f}배 (1.2배 넘으면 탈락)")

# Pivot 거리
pivot = recent['High'].iloc[last_peak_idx]
pivot_dist = (pivot - close) / close
cond_pivot = -0.05 <= pivot_dist <= 0.15

st.write(f"{'✅' if cond_pivot else '❌'} **Pivot 거리가 적당한가? (-5% ~ +15%)**")
st.caption(f"Pivot: {pivot:,.0f}원 / 현재가: {close:,.0f}원 / 거리: {pivot_dist*100:.1f}%")
