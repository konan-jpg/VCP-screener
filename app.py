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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_krx_stocks():
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq])
        
        stocks = stocks[~stocks['Name'].str.contains('우')]
        stocks = stocks[~stocks['Name'].str.contains('스팩')]
        
        if 'Marcap' in stocks.columns:
            stocks = stocks[stocks['Marcap'] >= 50_000_000_000]
            stocks = stocks.sort_values('Marcap', ascending=False)
        
        stocks['Marcap_billion'] = stocks['Marcap'] / 100_000_000
        return stocks[['Code', 'Name', 'Market', 'Marcap_billion']]
    except Exception as e:
        st.error(f"종목 로딩 실패: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_data(code, days=600):
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
        return df if len(df) > 0 else None
    except:
        return None

# -----------------------------------------------------------
# 2. 기술적 지표
# -----------------------------------------------------------
def ma(df, n):
    return df['Close'].rolling(n).mean()

def check_stage2_trend(df, min_price=10000):
    """Stage 2 확인 + 최소가격 필터"""
    if len(df) < 220:
        return False, "데이터 부족", None
    
    current_close = df['Close'].iloc[-1]
    
    # 최소 가격 체크
    if current_close < min_price:
        return False, f"가격 {current_close:,.0f}원 (최소 {min_price:,}원)", None

    ma_vals = {
        50: ma(df, 50),
        150: ma(df, 150),
        200: ma(df, 200)
    }
    
    m50 = ma_vals[50].iloc[-1]
    m150 = ma_vals[150].iloc[-1]
    m200 = ma_vals[200].iloc[-1]

    # 정배열 체크
    if not (current_close > m50 > m150 > m200):
        return False, "정배열 불량", None
    
    # 200일선 상승 추세
    m200_1m = ma_vals[200].iloc[-22]
    if m200 <= m200_1m:
        return False, "200일선 하락", None

    # 바닥 대비 상승
    low_52w = df['Low'].tail(252).min()
    rise = ((current_close - low_52w) / low_52w) * 100
    if rise < 30.0:
        return False, f"바닥 대비 {rise:.1f}%", None

    # 고점 대비 위치
    high_52w = df['High'].tail(252).max()
    if current_close < high_52w * 0.70:
        return False, "52주 고점 대비 낮음", None

    return True, "Stage 2 OK", ma_vals

# -----------------------------------------------------------
# 3. VCP 패턴 분석 (완화된 버전)
# -----------------------------------------------------------
def find_peaks_simple(series, distance=8):
    """고점 찾기"""
    peaks, _ = find_peaks(series.values, distance=distance)
    return peaks

def analyze_vcp_pattern(df, strictness='normal'):
    """
    VCP 패턴 분석
    strictness: 'strict' (엄격), 'normal' (보통), 'loose' (완화)
    """
    if df is None or len(df) < 100:
        return None, "데이터 부족"

    recent = df.tail(120).copy()
    recent['atr'] = (recent['High'] - recent['Low']) / recent['Close']
    
    # 1. 고점 찾기
    peaks_idx = find_peaks_simple(recent['High'], distance=8)
    
    if len(peaks_idx) < 2:
        return None, "파동 부족 (최소 2개 고점)"
    
    # 2. 각 파동별 변동성 계산
    waves = []
    for i in range(len(peaks_idx) - 1):
        start = peaks_idx[i]
        end = peaks_idx[i + 1]
        wave_vol = recent['atr'].iloc[start:end].mean()
        waves.append(wave_vol)
    
    # 마지막 파동 (핸들)
    last_peak_idx = peaks_idx[-1]
    handle_vol = recent['atr'].iloc[last_peak_idx:].mean()
    waves.append(handle_vol)
    
    # 최근 3개 파동만 사용
    recent_waves = waves[-3:] if len(waves) >= 3 else waves
    
    if len(recent_waves) < 2:
        return None, "분석 가능 파동 부족"
    
    # 3. 수축 패턴 검증 (완화 버전)
    # 엄격: 모든 파동이 순차 감소
    # 보통: 전체적으로 감소 추세 + 마지막이 가장 작음
    # 완화: 마지막이 첫 파동의 60% 이하면 OK
    
    if strictness == 'strict':
        # 모든 파동이 이전보다 작아야 함
        for i in range(len(recent_waves) - 1):
            if recent_waves[i] <= recent_waves[i + 1]:
                return None, f"파동 {i+1}→{i+2} 수축 실패"
    
    elif strictness == 'normal':
        # 마지막이 가장 작아야 하고, 첫 파동의 60% 이하
        if handle_vol >= min(recent_waves[:-1]):
            return None, "마지막 파동이 가장 작지 않음"
        
        if handle_vol > recent_waves[0] * 0.60:
            return None, f"수축 비율 부족 ({handle_vol/recent_waves[0]:.1%})"
    
    else:  # loose
        # 마지막이 첫 파동의 70% 이하면 OK
        if handle_vol > recent_waves[0] * 0.70:
            return None, f"수축 미흡 ({handle_vol/recent_waves[0]:.1%})"
    
    # 4. 절대 변동성 체크 (완화)
    max_handle_vol = {
        'strict': 0.035,  # 3.5%
        'normal': 0.06,   # 6.0% (수정: 두산로보틱스 4.56% 넉넉히 통과)
        'loose': 0.10     # 10.0% (수정: 가온전선 7.82% 넉넉히 통과)
    }[strictness]
    
    if handle_vol > max_handle_vol:
        return None, f"핸들 변동성 큼 ({handle_vol:.1%})"
    
    # 5. 거래량 분석 (완화)
    vol_ma50 = df['Volume'].rolling(50).mean().iloc[-1]
    handle_volume = recent['Volume'].iloc[last_peak_idx:].mean()
    
    vol_ratio_threshold = {
        'strict': 1.0, 
        'normal': 1.5,   # 1.5배 (수정: 두산로보틱스 1.41배 통과)
        'loose': 2.0     # 2.0배 (넉넉하게)
    }[strictness]
    
    vol_ratio = handle_volume / vol_ma50
    if vol_ratio > vol_ratio_threshold:
        return None, f"거래량 과다 ({vol_ratio:.1%})"
    
    # 6. Pivot 검증
    pivot = recent['High'].iloc[last_peak_idx]
    current_price = df['Close'].iloc[-1]
    
    days_since_pivot = len(recent) - last_peak_idx - 1
    if days_since_pivot > 35:
        return None, f"Pivot 후 {days_since_pivot}일 경과"
    
    pivot_dist_pct = ((pivot - current_price) / current_price) * 100
    
    if pivot_dist_pct < -3.0:
        return None, "이미 돌파 (진입 늦음)"
    
    max_pivot_dist = {
        'strict': 8.0,
        'normal': 12.0,
        'loose': 15.0
    }[strictness]
    
    if pivot_dist_pct > max_pivot_dist:
        return None, f"Pivot 거리 {pivot_dist_pct:.1f}%"
    
    # 7. 베이스 기간
    base_start = peaks_idx[0] if len(peaks_idx) > 0 else 0
    base_days = len(recent) - base_start
    
    if base_days < 15:
        return None, "베이스 너무 짧음"
    if base_days > 300:
        return None, "베이스 너무 김"
    
    return {
        "pivot": pivot,
        "handle_vol": handle_vol,
        "contraction_ratio": handle_vol / recent_waves[0],
        "volume_ratio": vol_ratio,
        "wave_count": len(peaks_idx),
        "base_days": base_days,
        "pivot_distance": pivot_dist_pct,
        "waves": recent_waves
    }, "VCP 확인"

# -----------------------------------------------------------
# 4. 자금 관리
# -----------------------------------------------------------
def calc_position(account, risk_pct, entry, stop_pct):
    risk_amt = account * (risk_pct / 100)
    stop = entry * (1 - stop_pct / 100)
    loss = entry - stop
    if loss <= 0:
        return stop, 0, 0, 0
    qty = int(risk_amt / loss)
    return stop, qty, qty * entry, (qty * entry / account) * 100

# -----------------------------------------------------------
# 5. 차트
# -----------------------------------------------------------
def plot_chart(df, name, code, pivot, stop, vcp_info):
    df_chart = df.tail(150)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name='Price'
    ), row=1, col=1)
    
    for period, color in [(50, 'blue'), (200, 'purple')]:
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=ma(df_chart, period),
            line=dict(color=color),
            name=f'{period}MA'
        ), row=1, col=1)
    
    fig.add_hline(y=pivot, line_dash='dash', line_color='green',
                  annotation_text=f'Pivot: {pivot:,.0f}', row=1, col=1)
    fig.add_hline(y=stop, line_dash='dot', line_color='red',
                  annotation_text=f'Stop: {stop:,.0f}', row=1, col=1)
    
    colors = ['red' if r.Open > r.Close else 'green' for r in df_chart.itertuples()]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'],
                         marker_color=colors), row=2, col=1)
    
    title = f"{name} ({code})"
    if vcp_info:
        title += f" | 수축: {vcp_info['contraction_ratio']:.1%} | 파동: {vcp_info['wave_count']}"
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# -----------------------------------------------------------
# 6. UI
# -----------------------------------------------------------
st.title("🦅 VCP Master Pro")
st.markdown("**미너비니 VCP 전략 | 우량주 중심 스크리너**")

with st.sidebar:
    st.header("⚙️ 설정")
    
    st.markdown("### 💰 자금 관리")
    account = st.number_input("총 자산 (원)", 10_000_000, 10_000_000_000, 50_000_000, 1_000_000)
    risk_pct = st.slider("리스크 (%)", 0.5, 2.5, 1.0, 0.1)
    stop_pct = st.slider("손절폭 (%)", 3.0, 8.0, 5.0, 0.5)
    
    st.divider()
    
    st.markdown("### 🔍 종목 필터")
    min_price = st.number_input("최소 주가 (원)", 5_000, 100_000, 10_000, 1_000)
    min_marcap = st.number_input("최소 시총 (억)", 100, 100_000, 2_000, 100)
    
    st.divider()
    
    st.markdown("### 🎯 VCP 엄격도")
    strictness = st.select_slider(
        "분석 기준",
        options=['strict', 'normal', 'loose'],
        value='normal',
        help="strict: 엄격 | normal: 보통 | loose: 완화"
    )
    
    strictness_desc = {
        'strict': "엄격 - 모든 파동 순차 감소 필수",
        'normal': "보통 - 전체적 감소 추세 + 마지막 최소",
        'loose': "완화 - 마지막이 첫 파동의 70% 이하"
    }
    st.caption(strictness_desc[strictness])
    
    st.divider()
    
    st.markdown("### 📊 스캔 설정")
    scan_count = st.selectbox(
        "스캔 종목 수",
        [100, 300, 500, 1000],
        index=1
    )
    
    if st.button("🚀 VCP 스캔 시작", type="primary", use_container_width=True):
        st.session_state['run'] = True
        st.session_state['candidates'] = []

if 'candidates' not in st.session_state:
    st.session_state['candidates'] = []

# -----------------------------------------------------------
# 7. 스캔 실행
# -----------------------------------------------------------
if st.session_state.get('run'):
    all_stocks = get_krx_stocks()
    
    if all_stocks.empty:
        st.error("종목 로딩 실패")
        st.session_state['run'] = False
    else:
        stocks_to_scan = all_stocks.head(scan_count)
        
        st.info(f"📊 시총 상위 {len(stocks_to_scan)}개 종목 분석 시작...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fail_stats = {}
        filtered_count = 0
        stage2_count = 0
        
        for idx, (_, row) in enumerate(stocks_to_scan.iterrows()):
            progress = (idx + 1) / len(stocks_to_scan)
            progress_bar.progress(progress)
            status_text.text(f"분석 중... {idx+1}/{len(stocks_to_scan)} - {row['Name']}")
            
            # 시총 필터
            if row['Marcap_billion'] < min_marcap:
                continue
            
            df = get_stock_data(row['Code'])
            if df is None:
                continue
            
            filtered_count += 1
            
            # Stage 2 체크 (최소 가격 포함)
            is_stage2, msg, _ = check_stage2_trend(df, min_price)
            if not is_stage2:
                fail_stats[msg] = fail_stats.get(msg, 0) + 1
                continue
            
            stage2_count += 1
            
            # VCP 분석
            vcp, vcp_msg = analyze_vcp_pattern(df, strictness)
            if vcp is None:
                fail_stats[vcp_msg] = fail_stats.get(vcp_msg, 0) + 1
                continue
            
            results.append({
                'Code': row['Code'],
                'Name': row['Name'],
                'Market': row['Market'],
                'Marcap': row['Marcap_billion'],
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
        with st.expander("📊 스캔 결과", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("시총 필터 통과", filtered_count)
            col2.metric("Stage 2", stage2_count)
            col3.metric("✅ VCP", len(results))
            col4.metric("발견율", f"{len(results)/filtered_count*100:.1f}%" if filtered_count > 0 else "0%")
            
            if fail_stats:
                st.markdown("**주요 탈락 사유**")
                sorted_fails = sorted(fail_stats.items(), key=lambda x: x[1], reverse=True)[:7]
                for reason, count in sorted_fails:
                    st.caption(f"• {reason}: {count}건")

# -----------------------------------------------------------
# 8. 결과 표시
# -----------------------------------------------------------
candidates = st.session_state['candidates']

if not candidates:
    st.info("👈 왼쪽에서 설정 후 스캔 시작")
else:
    st.success(f"✅ **{len(candidates)}개** VCP 후보!")
    
    with st.expander("📋 전체 리스트"):
        summary = pd.DataFrame([{
            '종목': c['Name'],
            '코드': c['Code'],
            '시총(억)': f"{c['Marcap']:,.0f}",
            '현재가': f"{c['Close']:,.0f}",
            '진입가': f"{c['Pivot']:,.0f}",
            '거리': f"{c['VCP']['pivot_distance']:.1f}%",
            '수축': f"{c['VCP']['contraction_ratio']:.1%}",
            '파동': c['VCP']['wave_count']
        } for c in candidates])
        st.dataframe(summary, use_container_width=True, hide_index=True)
    
    st.divider()
    
    selected = st.selectbox("상세 분석 종목", [c['Name'] for c in candidates])
    target = next(c for c in candidates if c['Name'] == selected)
    
    stop, qty, total, pos_pct = calc_position(account, risk_pct, target['Pivot'], stop_pct)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재가", f"{target['Close']:,.0f}원")
    col2.metric("진입가", f"{target['Pivot']:,.0f}원", f"{target['VCP']['pivot_distance']:+.1f}%")
    col3.metric("손절가", f"{stop:,.0f}원", f"-{stop_pct}%")
    col4.metric("수량", f"{qty:,}주", f"{pos_pct:.1f}%")
    
    fig = plot_chart(target['df'], target['Name'], target['Code'],
                     target['Pivot'], stop, target['VCP'])
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔬 VCP 상세"):
        vcp = target['VCP']
        st.write(f"- 파동 개수: {vcp['wave_count']}")
        st.write(f"- 수축 비율: {vcp['contraction_ratio']:.1%}")
        st.write(f"- 핸들 변동성: {vcp['handle_vol']:.2%}")
        st.write(f"- 거래량 비율: {vcp['volume_ratio']:.1%}")
        st.write(f"- 베이스 기간: {vcp['base_days']}일")
