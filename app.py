import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# -----------------------------------------------------------
# 1. 기본 설정
# -----------------------------------------------------------
st.set_page_config(page_title="VCP Master Pro", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_krx_stocks():
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq])
        # 우선주 제외
        stocks = stocks[~stocks['Name'].str.contains('우')]
        return stocks[['Code', 'Name', 'Market']]
    except Exception as e:
        st.error(f"종목 리스트 로딩 실패: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_data(code, days=600):
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
        return df if len(df) > 0 else None
    except Exception:
        return None

# -----------------------------------------------------------
# 2. 기술적 지표
# -----------------------------------------------------------
def ma(df, n):
    """이동평균선 계산"""
    return df['Close'].rolling(n).mean()

def check_stage2_trend(df):
    """
    마크 미너비니 Trend Template (완벽 구현)
    """
    if len(df) < 260:
        return False, "데이터 부족", None

    # 이동평균선 계산 (재사용을 위해 dict로 반환)
    ma_values = {
        50: ma(df, 50),
        150: ma(df, 150),
        200: ma(df, 200)
    }
    
    current_close = df['Close'].iloc[-1]
    ma50_now = ma_values[50].iloc[-1]
    ma150_now = ma_values[150].iloc[-1]
    ma200_now = ma_values[200].iloc[-1]

    # [조건 1] 완전한 정배열: 현재가 > 50 > 150 > 200
    if not (current_close > ma50_now > ma150_now > ma200_now):
        if current_close < ma200_now:
            return False, "현재가 < 200일선", None
        elif current_close < ma150_now:
            return False, "현재가 < 150일선", None
        elif current_close < ma50_now:
            return False, "현재가 < 50일선", None
        elif ma50_now < ma150_now:
            return False, "50일선 < 150일선", None
        elif ma150_now < ma200_now:
            return False, "150일선 < 200일선", None
        return False, "정배열 불량", None

    # [조건 2] 200일선 상승 추세 (1개월 + 3개월)
    ma200_1m = ma_values[200].iloc[-22]
    ma200_3m = ma_values[200].iloc[-66]
    
    if ma200_now <= ma200_1m:
        return False, "200일선 1개월간 미상승", None
    if ma200_now <= ma200_3m:
        return False, "200일선 3개월간 미상승", None
    
    # 200일선 기울기
    slope_200 = (ma200_now - ma200_3m) / ma200_3m
    if slope_200 < 0.03:
        return False, f"200일선 기울기 부족 ({slope_200*100:.1f}%)", None

    # [조건 3] 50일선 상승 추세
    ma50_2w = ma_values[50].iloc[-10]
    if ma50_now <= ma50_2w:
        return False, "50일선 하락/횡보", None

    # [조건 4] 현재가가 50일선 매우 근처 (VCP는 50일선 위에서 형성)
    dist_ma50 = ((current_close - ma50_now) / ma50_now) * 100
    if dist_ma50 < -3.0:  # -3% 이내로 엄격하게
        return False, f"50일선 대비 {dist_ma50:.1f}% 이탈", None

    # [조건 5] 52주 최저가 대비 상승폭
    low_52w = df['Low'].tail(252).min()
    rise_from_low = ((current_close - low_52w) / low_52w) * 100
    if rise_from_low < 40.0:
        return False, f"52주 최저 대비 {rise_from_low:.1f}% (40% 미만)", None

    # [조건 6] 52주 최고가 대비 위치
    high_52w = df['High'].tail(252).max()
    if current_close < high_52w * 0.75:
        dist_from_high = ((current_close - high_52w) / high_52w) * 100
        return False, f"52주 최고가 대비 {dist_from_high:.1f}%", None

    # [조건 7] 200일선 대비 충분한 상승
    dist_ma200 = ((current_close - ma200_now) / ma200_now) * 100
    if dist_ma200 < 15.0:
        return False, f"200일선 대비 {dist_ma200:.1f}% (부족)", None

    return True, "Stage 2 확인", ma_values

# -----------------------------------------------------------
# 3. VCP 패턴 분석 (완전 개선)
# -----------------------------------------------------------
def find_local_peaks_and_troughs(series, distance=5):
    """고점과 저점 모두 찾기"""
    peaks, _ = find_peaks(series.values, distance=distance)
    troughs, _ = find_peaks(-series.values, distance=distance)
    return peaks, troughs

def analyze_vcp_pattern(df):
    """
    VCP 패턴 정밀 분석 - 완전 개선판
    """
    if df is None or len(df) < 120:
        return None, "데이터 부족"

    # 최근 100일로 베이스 분석
    recent = df.tail(100).copy()
    
    # 1. ATR 기반 변동성 계산
    recent['atr'] = (recent['High'] - recent['Low']) / recent['Close']
    
    # 2. 고점과 저점 찾기
    peaks_idx, troughs_idx = find_local_peaks_and_troughs(recent['Close'], distance=5)
    
    if len(peaks_idx) < 2:
        return None, "파동 부족 (최소 2개 고점 필요)"
    
    # 3. 파동 구간 정의 (고점 → 저점 → 고점)
    # 각 파동 = 한 고점에서 다음 고점까지
    waves = []
    for i in range(len(peaks_idx) - 1):
        wave_start = peaks_idx[i]
        wave_end = peaks_idx[i + 1]
        
        # 이 파동의 평균 변동성
        wave_volatility = recent['atr'].iloc[wave_start:wave_end].mean()
        
        # 이 파동의 가격 하락폭 (조정 깊이)
        peak_price = recent['Close'].iloc[wave_start]
        # 이 구간의 최저가 찾기
        trough_price = recent['Close'].iloc[wave_start:wave_end].min()
        pullback_pct = ((peak_price - trough_price) / peak_price) * 100
        
        waves.append({
            'volatility': wave_volatility,
            'pullback': pullback_pct,
            'start_idx': wave_start,
            'end_idx': wave_end
        })
    
    # 마지막 파동 (현재 진행 중)
    last_peak_idx = peaks_idx[-1]
    current_wave_volatility = recent['atr'].iloc[last_peak_idx:].mean()
    last_peak_price = recent['Close'].iloc[last_peak_idx]
    current_trough = recent['Close'].iloc[last_peak_idx:].min()
    current_pullback = ((last_peak_price - current_trough) / last_peak_price) * 100
    
    waves.append({
        'volatility': current_wave_volatility,
        'pullback': current_pullback,
        'start_idx': last_peak_idx,
        'end_idx': len(recent) - 1
    })
    
    # 최근 3~4개 파동만 사용
    recent_waves = waves[-4:] if len(waves) >= 4 else waves[-3:]
    
    if len(recent_waves) < 3:
        return None, f"분석 가능 파동 {len(recent_waves)}개 부족"
    
    # 4. 수축 패턴 검증 (각 파동이 이전보다 작아야 함)
    volatilities = [w['volatility'] for w in recent_waves]
    pullbacks = [w['pullback'] for w in recent_waves]
    
    # 변동성이 점진적으로 감소하는지
    for i in range(len(volatilities) - 1):
        if volatilities[i] <= volatilities[i + 1]:
            return None, f"파동 {i+1}→{i+2} 수축 실패: {volatilities[i]:.3f}→{volatilities[i+1]:.3f}"
    
    # 조정폭도 점진적으로 감소해야 함
    for i in range(len(pullbacks) - 1):
        if pullbacks[i] <= pullbacks[i + 1]:
            return None, f"조정폭 {i+1}→{i+2} 수축 실패: {pullbacks[i]:.1f}%→{pullbacks[i+1]:.1f}%"
    
    # 수축 비율 체크
    contraction_ratio = volatilities[-1] / volatilities[0]
    if contraction_ratio > 0.50:
        return None, f"수축 비율 부족 ({contraction_ratio:.1%})"
    
    # 마지막 파동의 절대적 변동성 체크
    if volatilities[-1] > 0.04:  # 4% 이상이면 너무 넓음
        return None, f"마지막 파동 변동성 과다 ({volatilities[-1]:.1%})"
    
    # 5. 거래량 Dry-up 검증
    vol_ma50 = df['Volume'].rolling(50).mean().iloc[-1]
    
    # 각 파동별 평균 거래량
    wave_volumes = []
    for wave in recent_waves:
        wave_vol = recent['Volume'].iloc[wave['start_idx']:wave['end_idx']].mean()
        wave_volumes.append(wave_vol / vol_ma50)  # 정규화
    
    # 거래량도 점진적으로 감소해야 함
    if not all(wave_volumes[i] > wave_volumes[i+1] for i in range(len(wave_volumes)-1)):
        return None, f"거래량 미감소: {[f'{v:.2f}x' for v in wave_volumes]}"
    
    # 마지막 파동 거래량이 평균의 70% 이하
    if wave_volumes[-1] > 0.70:
        return None, f"마지막 파동 거래량 과다 ({wave_volumes[-1]:.1%})"
    
    # 6. Pivot 설정 및 검증
    pivot_price = recent['Close'].iloc[last_peak_idx]
    current_price = df['Close'].iloc[-1]
    
    # Pivot 시간 경과
    days_since_pivot = len(recent) - last_peak_idx - 1
    if days_since_pivot > 30:
        return None, f"Pivot 후 {days_since_pivot}일 경과 (너무 오래됨)"
    
    # 현재가 vs Pivot 거리
    pivot_dist = ((pivot_price - current_price) / current_price) * 100
    
    if pivot_dist < 0:
        # 이미 돌파
        if current_price > pivot_price * 1.03:
            return None, "Pivot 3% 이상 돌파 (진입 시점 놓침)"
    elif pivot_dist > 8.0:
        return None, f"Pivot 거리 {pivot_dist:.1f}% (너무 멀음)"
    
    # 7. 베이스 기간 검증
    base_start_idx = peaks_idx[0] if len(peaks_idx) > 0 else 0
    base_days = len(recent) - base_start_idx
    
    if base_days < 21:  # 3주 미만
        return None, f"베이스 {base_days}일 (너무 짧음)"
    if base_days > 250:  # 1년 초과
        return None, f"베이스 {base_days}일 (너무 김)"
    
    # 8. 추가 검증: 현재가가 베이스 중간 이상에 위치
    base_high = recent['High'].iloc[base_start_idx:].max()
    base_low = recent['Low'].iloc[base_start_idx:].min()
    base_position = (current_price - base_low) / (base_high - base_low)
    
    if base_position < 0.60:  # 베이스 하단 40%에 있으면 위험
        return None, f"베이스 하단에 위치 ({base_position:.1%})"
    
    return {
        "pivot": pivot_price,
        "contraction_ratio": contraction_ratio,
        "volume_ratio": wave_volumes[-1],
        "wave_count": len(recent_waves),
        "base_days": base_days,
        "pivot_distance": pivot_dist,
        "volatilities": volatilities,
        "pullbacks": pullbacks,
        "wave_volumes": wave_volumes,
        "base_position": base_position
    }, "VCP 패턴 확인"

# -----------------------------------------------------------
# 4. 자금 관리
# -----------------------------------------------------------
def calculate_position_sizing(account, risk_pct, entry, stop_pct):
    """포지션 사이징 계산"""
    risk_amount = account * (risk_pct / 100)
    stop_price = entry * (1 - stop_pct / 100)
    loss_per_share = entry - stop_price

    if loss_per_share <= 0:
        return stop_price, 0, 0, 0.0

    qty = int(risk_amount / loss_per_share)
    total_invest = qty * entry
    position_pct = (total_invest / account) * 100

    return stop_price, qty, total_invest, position_pct

# -----------------------------------------------------------
# 5. 차트
# -----------------------------------------------------------
def plot_chart(df, code, name, pivot, stop, vcp_info=None):
    """차트 시각화"""
    df_chart = df.tail(150)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03
    )

    # 캔들
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'], high=df_chart['High'],
        low=df_chart['Low'], close=df_chart['Close'],
        name='Price'
    ), row=1, col=1)

    # 이평선
    for period, color, width in [(50, 'blue', 2), (150, 'green', 1), (200, 'purple', 1)]:
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=ma(df_chart, period),
            line=dict(color=color, width=width),
            name=f'{period}MA'
        ), row=1, col=1)

    # Pivot & Stop
    fig.add_hline(
        y=pivot, line_dash='dash', line_color='green', line_width=2,
        annotation_text=f'🎯 Pivot: {pivot:,.0f}',
        annotation_position="right", row=1, col=1
    )
    
    fig.add_hline(
        y=stop, line_dash='dot', line_color='red', line_width=2,
        annotation_text=f'🛑 Stop: {stop:,.0f}',
        annotation_position="right", row=1, col=1
    )

    # 거래량
    colors = ['red' if r.Open > r.Close else 'green' for r in df_chart.itertuples()]
    fig.add_trace(go.Bar(
        x=df_chart.index, y=df_chart['Volume'],
        marker_color=colors, name='Volume'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=df_chart['Volume'].rolling(50).mean(),
        line=dict(color='orange', dash='dash'),
        name='Vol 50MA'
    ), row=2, col=1)

    title = f"{name} ({code})"
    if vcp_info:
        title += f" | 수축: {vcp_info['contraction_ratio']:.1%} | 파동: {vcp_info['wave_count']} | 베이스: {vcp_info['base_days']}일"
    
    fig.update_layout(
        title=title,
        height=650,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# -----------------------------------------------------------
# 6. UI
# -----------------------------------------------------------
st.title("🦅 VCP Master Pro (최종 검증판)")
st.markdown("**마크 미너비니 VCP 전략 완벽 구현** | Stage 2 + 파동 수축 + 거래량 Dry-up + 자금 관리")

with st.sidebar:
    st.header("⚙️ 설정")
    
    st.markdown("### 💰 자금 관리")
    account = st.number_input("총 자산 (원)", 10_000_000, 10_000_000_000, 50_000_000, 1_000_000)
    risk_pct = st.slider("계좌 리스크 (%)", 0.5, 2.5, 1.0, 0.1)
    stop_pct = st.slider("손절폭 (%)", 3.0, 8.0, 5.0, 0.5)
    
    max_loss = account * risk_pct / 100
    st.info(f"💡 1회 최대 손실: **{max_loss:,.0f}원**")
    
    st.divider()
    
    st.markdown("### 🔍 스캔 설정")
    top_n = st.number_input("스캔 종목 수", 20, 1000, 100, 10)
    
    if st.button("🚀 VCP 스캔 시작", type="primary", use_container_width=True):
        st.session_state['run'] = True
        st.session_state['candidates'] = []

if 'candidates' not in st.session_state:
    st.session_state['candidates'] = []

# -----------------------------------------------------------
# 7. 스캔 실행
# -----------------------------------------------------------
if st.session_state.get('run'):
    stocks = get_krx_stocks().head(top_n)
    
    if stocks.empty:
        st.error("종목 데이터 로딩 실패")
        st.session_state['run'] = False
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fail_stats = {}
        stage2_count = 0
        
        for idx, (_, row) in enumerate(stocks.iterrows()):
            progress = (idx + 1) / len(stocks)
            progress_bar.progress(progress)
            status_text.text(f"분석 중... {idx+1}/{len(stocks)} - {row['Name']}")
            
            df = get_stock_data(row['Code'])
            if df is None:
                continue
            
            # Stage 2 체크
            is_stage2, msg, _ = check_stage2_trend(df)
            if not is_stage2:
                fail_stats[msg] = fail_stats.get(msg, 0) + 1
                continue
            
            stage2_count += 1
            
            # VCP 분석
            vcp, vcp_msg = analyze_vcp_pattern(df)
            if vcp is None:
                fail_stats[vcp_msg] = fail_stats.get(vcp_msg, 0) + 1
                continue
            
            results.append({
                'Code': row['Code'],
                'Name': row['Name'],
                'Market': row['Market'],
                'Close': df['Close'].iloc[-1],
                'Pivot': vcp['pivot'],
                'VCP': vcp,
                'df': df
            })
        
        st.session_state['candidates'] = results
        st.session_state['run'] = False
        
        progress_bar.empty()
        status_text.empty()
        
        # 통계
        with st.expander("📊 스캔 결과 통계", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("전체 스캔", len(stocks))
            col2.metric("Stage 2 통과", stage2_count)
            col3.metric("✅ VCP 발견", len(results))
            
            if fail_stats:
                st.markdown("**주요 탈락 사유 (상위 5개)**")
                sorted_fails = sorted(fail_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                for reason, count in sorted_fails:
                    st.caption(f"• {reason}: {count}건")

# -----------------------------------------------------------
# 8. 결과 표시
# -----------------------------------------------------------
candidates = st.session_state['candidates']

if not candidates:
    st.info("👈 왼쪽에서 스캔을 시작하세요")
    
    with st.expander("💡 VCP 패턴 학습 자료"):
        st.markdown("""
        ### Volatility Contraction Pattern (VCP)
        
        **핵심 개념**:
        - 3~4개의 연속된 조정 파동
        - 각 파동의 **변동성**과 **조정폭**이 점진적으로 감소
        - **거래량**도 파동마다 감소 (Dry-up)
        - Stage 2 상승 추세 중에 발생
        
        **진입 규칙**:
        1. Pivot(마지막 고점) 돌파 확인
        2. 돌파 시 거래량 평균 대비 **40~50% 증가** 필수
        3. 당일 또는 익일 재진입 시점에서 매수
        
        **손절 규칙**:
        - Pivot 기준 **5~7% 하락** 시 무조건 청산
        - 예외 없음
        
        **익절 전략**:
        - +20%: 1/3 익절
        - +40%: 추가 1/3 익절
        - 나머지: 50일선 이탈 시 전량 청산
        """)
else:
    st.success(f"✅ **{len(candidates)}개** VCP 후보 발견!")
    
    # 요약 테이블
    with st.expander("📋 전체 후보 리스트", expanded=False):
        summary_df = pd.DataFrame([{
            '종목명': c['Name'],
            '코드': c['Code'],
            '시장': c['Market'],
            '현재가': f"{c['Close']:,.0f}",
            '진입가': f"{c['Pivot']:,.0f}",
            '거리': f"{c['VCP']['pivot_distance']:.1f}%",
            '수축비': f"{c['VCP']['contraction_ratio']:.1%}",
            '파동': c['VCP']['wave_count'],
            '베이스': f"{c['VCP']['base_days']}일"
        } for c in candidates])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 상세 분석
    st.subheader("🎯 종목 상세 분석")
    
    selected_name = st.selectbox("분석할 종목 선택", [c['Name'] for c in candidates])
    target = next(c for c in candidates if c['Name'] == selected_name)
    
    # 포지션 사이징
    stop, qty, total, pos_pct = calculate_position_sizing(
        account, risk_pct, target['Pivot'], stop_pct
    )
    
    # 주요 지표
    st.markdown("### 📊 매매 전략")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("현재가", f"{target['Close']:,.0f}원")
    col2.metric("🎯 진입가", f"{target['Pivot']:,.0f}원",
                f"{target['VCP']['pivot_distance']:+.1f}%")
    col3.metric("🛑 손절가", f"{stop:,.0f}원", f"-{stop_pct}%")
    col4.metric("매수 수량", f"{qty:,}주")
    col5.metric("투입 금액", f"{total:,.0f}원", f"{pos_pct:.1f}%")
    
    # 경고
    if pos_pct > 20:
        st.error(f"⚠️ 비중 {pos_pct:.1f}%는 과도합니다! 손절폭을 줄이거나 리스크를 낮추세요.")
    elif pos_pct > 15:
        st.warning(f"⚠️ 비중 {pos_pct:.1f}%는 다소 높습니다. 15% 이하 권장")
    
    # 차트
    st.markdown("### 📈 차트 분석")
    fig = plot_chart(target['df'], target['Code'], target['Name'],
                     target['Pivot'], stop, target['VCP'])
    st.plotly_chart(fig, use_container_width=True)
    
    # VCP 상세 정보
    with st.expander("🔬 VCP 패턴 상세 분석", expanded=True):
        vcp = target['VCP']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("파동 개수", f"{vcp['wave_count']}개")
        col2.metric("수축 비율", f"{vcp['contraction_ratio']:.1%}")
        col3.metric("거래량 비율", f"{vcp['volume_ratio']:.1%}")
        col4.metric("베이스 기간", f"{vcp['base_days']}일")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**파동별 변동성**")
            for i, v in enumerate(reversed(vcp['volatilities']), 1):
                wave_num = len(vcp['volatilities']) - i + 1
                st.text(f"파동 {wave_num}: {v:.2%}")
        
        with col2:
            st.markdown("**파동별 조정폭**")
            for i, p in enumerate(reversed(vcp['pullbacks']), 1):
                wave_num = len(vcp['pullbacks']) - i + 1
                st.text(f"파동 {wave_num}: {p:.1f}%")
        
        st.markdown("---")
        
        # 품질 평가
        quality_score = 0
        if vcp['contraction_ratio'] < 0.40:
            quality_score += 1
        if vcp['volume_ratio'] < 0.50:
            quality_score += 1
        if vcp['base_position'] > 0.75:
            quality_score += 1
        if 30 <= vcp['base_days'] <= 120:
            quality_score += 1
        
        quality_text = ["불량", "보통", "양호", "우수", "최우수"][quality_score]
        quality_color = ["🔴", "🟡", "🟢", "🟢", "🟢"][quality_score]
        
        st.info(f"""
        **VCP 품질 평가**: {quality_color} **{quality_text}** ({quality_score}/4점)
        
        - 수축 비율: {'✅' if vcp['contraction_ratio'] < 0.40 else '⚠️'} {vcp['contraction_ratio']:.1%}
        - 거래량 감소: {'✅' if vcp['volume_ratio'] < 0.50 else '⚠️'} {vcp['volume_ratio']:.1%}
        - 베이스 위치: {'✅' if vcp['base_position'] > 0.75 else '⚠️'} 상위 {(1-vcp['base_position'])*100:.0f}%
        - 베이스 기간: {'✅' if 30 <= vcp['base_days'] <= 120 else '⚠️'} {vcp['base_days']}일
        """)
    
    # 매매 가이드
    st.markdown("---")
    st.markdown("### 📋 실전 매매 가이드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **🟢 진입 조건 (모두 충족 시)**
        
        1. 가격이 **{target['Pivot']:,.0f}원** 돌파
        2. 돌파 시 거래량 **평균 대비 40% 이상** 증가
        3. 장중 돌파: 당일 종가 매수
        4. 장 마감 후 돌파: 익일 재진입 확인 후 매수
        5. 갭 상승 돌파: 당일 고가 대비 -2% 이내 진입
        """)
    
    with col2:
        st.markdown(f"""
        **🔴 손절 / 익절 규칙**
        
        - **손절**: {stop:,.0f}원 ({stop_pct}%) 이탈 시 즉시 청산
        - **1차 익절**: +20% → 30% 물량 익절
        - **2차 익절**: +40% → 추가 30% 익절
        - **최종 청산**: 50일선 -3% 이탈 시 전량 청산
        - **예외 없음**: 손절가는 절대 지켜야 함
        """)
    
    st.warning("""
    ⚠️ **필수 체크리스트**
    - [ ] 진입 전 뉴스/공시 확인 (이벤트성 급등은 제외)
    - [ ] 거래량 증가 반드시 확인
    - [ ] 손절가 미리 설정 (지정가 주문)
    - [ ] 포지션 비중 15% 이하 유지
    - [ ] 감정적 판단 배제 (기계적 실행)
    
    💡 **이 도구는 보조 수단**이며, 최종 투자 결정은 본인의 책임입니다.
    """)
