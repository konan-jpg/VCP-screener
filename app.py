import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------
# 1. 기본 설정
# -----------------------------------------------------------
st.set_page_config(page_title="VCP Tightness Scanner v3", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .bonus-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_krx_stocks():
    """시총 2,000억 이상 종목만"""
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq])
        
        stocks = stocks[~stocks['Name'].str.contains('우')]
        stocks = stocks[~stocks['Name'].str.contains('스팩')]
        
        # ✅ 시총 2,000억 이상 (패턴 신뢰성 하한선)
        if 'Marcap' in stocks.columns:
            stocks = stocks[stocks['Marcap'] >= 200_000_000_000]
            stocks = stocks.sort_values('Marcap', ascending=False)
            stocks['Marcap_billion'] = stocks['Marcap'] / 100_000_000
        
        return stocks[['Code', 'Name', 'Market', 'Marcap_billion']]
    except Exception as e:
        st.error(f"종목 로딩 실패: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_data(code, days=200):
    """주식 데이터"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 0 else None
    except:
        return None

# -----------------------------------------------------------
# 2. VCP Tightness Scanner v3 (최종)
# -----------------------------------------------------------
def vcp_tightness_scanner(df, short_period=10, long_period=60, atr_period=20):
    """
    VCP Tightness Scanner v3 - 완전판
    
    핵심 개선:
    1. 시총 2,000억 이상 (하드 필터)
    2. 현재가 10,000원 이상 (하드 필터)
    3. 저점 유지력 보너스
    4. 조용한 양봉 연속성 (최대 3일)
    """
    if df is None or len(df) < long_period + atr_period:
        return None
    
    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # ✅ 현재가 10,000원 이상 (통계적 왜곡 제거)
    current_price = close.iloc[-1]
    if current_price < 10_000:
        return None
    
    # -----------------------
    # 1. Price Tightness
    # -----------------------
    std_price_short = close.tail(short_period).std()
    std_price_long = close.tail(long_period).std()
    
    if std_price_long == 0 or pd.isna(std_price_long):
        return None
    
    price_tightness = std_price_short / std_price_long
    
    # -----------------------
    # 2. Volume Dry-up
    # -----------------------
    std_vol_short = volume.tail(short_period).std()
    std_vol_long = volume.tail(long_period).std()
    
    if std_vol_long == 0 or pd.isna(std_vol_long):
        return None
    
    volume_dryup = std_vol_short / std_vol_long
    
    # -----------------------
    # 3. Range Contraction
    # -----------------------
    range_pct = (high - low) / close
    range_short = range_pct.tail(short_period).mean()
    range_long = range_pct.tail(long_period).mean()
    
    if range_long == 0 or pd.isna(range_long):
        return None
    
    range_ratio = range_short / range_long
    
    # -----------------------
    # 4. ATR 계산
    # -----------------------
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        return None
    
    # -----------------------
    # 5. 조용한 양봉 연속성 (최대 3일)
    # -----------------------
    quiet_days = 0
    for i in range(1, 4):  # 최근 3일
        if len(close) < i:
            break
        
        day_close = close.iloc[-i]
        day_open = open_.iloc[-i]
        body = abs(day_close - day_open)
        
        # 양봉 + 몸통이 ATR의 40% 이하
        if day_close > day_open and body <= atr * 0.40:
            quiet_days += 1
    
    # 누적 보너스 (1일당 5%, 최대 15%)
    quiet_bonus = 1.0 - min(quiet_days * 0.05, 0.15)
    
    # -----------------------
    # 6. 저점 유지력 보너스
    # -----------------------
    recent_low = low.tail(short_period).min()
    long_low = low.tail(long_period).min()
    
    # 최근 저점이 장기 저점의 101% 이상 유지 시 보너스
    low_hold = recent_low >= long_low * 1.01
    low_hold_bonus = 0.90 if low_hold else 1.0
    
    # -----------------------
    # 7. 기본 점수 계산
    # -----------------------
    base_score = (
        price_tightness * 0.50 +
        volume_dryup * 0.30 +
        range_ratio * 0.20
    )
    
    # -----------------------
    # 8. 최종 점수 (보너스 적용)
    # -----------------------
    final_score = base_score * quiet_bonus * low_hold_bonus
    
    return {
        "score": final_score,
        "base_score": base_score,
        "price_tightness": price_tightness,
        "volume_dryup": volume_dryup,
        "range_ratio": range_ratio,
        "quiet_days": quiet_days,
        "quiet_bonus": quiet_bonus,
        "low_hold": low_hold,
        "low_hold_bonus": low_hold_bonus,
        "atr": atr,
        "current_price": current_price,
        "recent_low": recent_low,
        "long_low": long_low
    }

# -----------------------------------------------------------
# 3. 차트
# -----------------------------------------------------------
def plot_chart(df, name, code, result):
    """차트 시각화"""
    df_chart = df.tail(120)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    # 캔들
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name='Price'
    ), row=1, col=1)
    
    # 50일선 (참고용)
    ma50 = df_chart['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=ma50,
        line=dict(color='blue', width=1, dash='dot'),
        name='50MA (참고)'
    ), row=1, col=1)
    
    # 거래량
    colors = ['red' if r.Open > r.Close else 'green' for r in df_chart.itertuples()]
    fig.add_trace(go.Bar(
        x=df_chart.index,
        y=df_chart['Volume'],
        marker_color=colors
    ), row=2, col=1)
    
    title = f"{name} ({code})"
    if result:
        title += f" | 점수: {result['score']:.3f} | 조용한양봉: {result['quiet_days']}일"
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# -----------------------------------------------------------
# 4. UI
# -----------------------------------------------------------
st.title("🔍 VCP Tightness Scanner v3 (최종)")
st.markdown("""
**완성된 VCP 스캐너 - 4가지 핵심 개선**

✅ **하드 필터**:
- 시총 2,000억 이상 (패턴 신뢰성)
- 현재가 10,000원 이상 (통계적 의미)

✅ **보너스 시스템**:
- 저점 유지력: 10% 감소
- 조용한 양봉 연속: 최대 15% 감소

✅ **철학**:
- VCP 판별 ❌ → 랭킹 ⭕
- 절대 기준 ❌ → 상대 평가 ⭕
""")

with st.sidebar:
    st.header("⚙️ 설정")
    
    st.markdown("### 📊 스캔 대상")
    scan_count = st.selectbox(
        "시총 상위 N개",
        [100, 300, 500, 1000],
        index=1
    )
    
    st.caption("※ 이미 시총 2,000억 이상만 포함됨")
    
    st.divider()
    
    st.markdown("### 🔬 파라미터")
    short_period = st.slider("단기 (일)", 5, 20, 10, 1)
    long_period = st.slider("장기 (일)", 40, 120, 60, 5)
    atr_period = st.slider("ATR 기간", 10, 30, 20, 5)
    
    st.divider()
    
    st.markdown("### 🎯 결과")
    top_n = st.slider("상위 표시", 10, 100, 30, 5)
    
    st.divider()
    
    if st.button("🚀 스캔 시작", type="primary", use_container_width=True):
        st.session_state['run'] = True
        st.session_state['results'] = []

if 'results' not in st.session_state:
    st.session_state['results'] = []

# -----------------------------------------------------------
# 5. 스캔 실행
# -----------------------------------------------------------
if st.session_state.get('run'):
    stocks = get_krx_stocks()
    
    if stocks.empty:
        st.error("종목 로딩 실패")
        st.session_state['run'] = False
    else:
        stocks_to_scan = stocks.head(scan_count)
        
        st.info(f"📊 시총 2,000억+ 상위 {len(stocks_to_scan)}개 스캔 중...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(stocks_to_scan.iterrows()):
            progress = (idx + 1) / len(stocks_to_scan)
            progress_bar.progress(progress)
            status_text.text(f"{idx+1}/{len(stocks_to_scan)} - {row['Name']}")
            
            df = get_stock_data(row['Code'])
            if df is None:
                continue
            
            result = vcp_tightness_scanner(df, short_period, long_period, atr_period)
            
            if result is not None:
                results.append({
                    'Code': row['Code'],
                    'Name': row['Name'],
                    'Market': row['Market'],
                    'Marcap': row['Marcap_billion'],
                    'df': df,
                    **result
                })
        
        progress_bar.empty()
        status_text.empty()
        
        if len(results) == 0:
            st.warning("조건에 맞는 종목 없음")
            st.session_state['run'] = False
        else:
            ranking = pd.DataFrame(results).sort_values('score').head(top_n)
            st.session_state['results'] = ranking.to_dict('records')
            st.session_state['run'] = False
            
            st.success(f"✅ 완료! {len(results)}개 중 상위 {len(ranking)}개")

# -----------------------------------------------------------
# 6. 결과 표시
# -----------------------------------------------------------
results = st.session_state['results']

if not results:
    st.info("👈 설정 후 스캔 시작")
    
    with st.expander("💡 v3 개선 사항"):
        st.markdown("""
        ### 왜 이 4가지가 필수인가?
        
        #### 1. 시총 2,000억 이상
        - 소형주: 세력 1~2명으로 패턴 왜곡
        - 중대형주: 기관/외국인 자금 = 진짜 패턴
        
        #### 2. 현재가 10,000원 이상
        - 저가주: 호가 단위 영향 큼 → 통계 왜곡
        - 10,000원+: 통계적 의미 있음
        
        #### 3. 저점 유지력 보너스 (10%)
        - 문제: 죽은 종목도 조용함
        - 해결: 저점 지키면서 조용한지 확인
        - VCP = 조정 (하락 아님)
        
        #### 4. 조용한 양봉 연속성 (최대 15%)
        - 1일: 우연일 수 있음
        - 2~3일 연속: 신뢰도 급상승
        - VCP 핸들 = 조용한 양봉 반복
        
        ### 점수 계산:
        ```
        기본 = (가격조임×0.5 + 거래량×0.3 + 레인지×0.2)
        최종 = 기본 × 조용한양봉보너스 × 저점유지보너스
        
        예시:
        기본 0.50
        → 조용한 양봉 3일 (×0.85)
        → 저점 유지 (×0.90)
        = 0.50 × 0.85 × 0.90 = 0.38
        ```
        """)
else:
    st.success(f"🎯 가장 조여진 상위 {len(results)}개")
    
    # 요약 테이블
    with st.expander("📋 전체 랭킹", expanded=True):
        summary = pd.DataFrame([{
            '순위': idx + 1,
            '종목': r['Name'],
            '시총(억)': f"{r['Marcap']:,.0f}",
            '현재가': f"{r['current_price']:,.0f}",
            '점수': f"{r['score']:.3f}",
            '기본': f"{r['base_score']:.3f}",
            '조용한양봉': f"{r['quiet_days']}일",
            '저점유지': '✅' if r['low_hold'] else '❌',
            '가격조임': f"{r['price_tightness']:.3f}",
            '거래량': f"{r['volume_dryup']:.3f}"
        } for idx, r in enumerate(results)])
        
        st.dataframe(summary, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 상세 분석
    st.subheader("📊 상세 분석")
    
    selected = st.selectbox(
        "종목 선택",
        [f"{idx+1}위. {r['Name']} - {r['score']:.3f}" 
         for idx, r in enumerate(results)]
    )
    
    selected_idx = int(selected.split('위')[0]) - 1
    target = results[selected_idx]
    
    # 지표
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("순위", f"{selected_idx + 1}위")
    col2.metric("최종점수", f"{target['score']:.3f}")
    col3.metric("기본점수", f"{target['base_score']:.3f}")
    col4.metric("조용한양봉", f"{target['quiet_days']}일")
    col5.metric("저점유지", "✅" if target['low_hold'] else "❌")
    
    # 보너스 상세
    st.markdown(
        f'<div class="bonus-box">'
        f'<b>보너스 적용 내역</b><br>'
        f'• 기본 점수: {target["base_score"]:.3f}<br>'
        f'• 조용한 양봉 보너스: ×{target["quiet_bonus"]:.2f} ({target["quiet_days"]}일 연속)<br>'
        f'• 저점 유지 보너스: ×{target["low_hold_bonus"]:.2f} '
        f'(최근저점 {target["recent_low"]:,.0f} vs 장기저점 {target["long_low"]:,.0f})<br>'
        f'• <b>최종 점수: {target["score"]:.3f}</b>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # 차트
    fig = plot_chart(target['df'], target['Name'], target['Code'], target)
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 지표
    with st.expander("🔬 상세 지표"):
        st.markdown(f"""
        ### {target['Name']} 상세 분석
        
        **최종 점수: {target['score']:.3f}**
        
        #### 점수 구성:
        - 기본: {target['base_score']:.3f}
        - 조용한 양봉: ×{target['quiet_bonus']:.2f} ({target['quiet_days']}일)
        - 저점 유지: ×{target['low_hold_bonus']:.2f}
        
        #### 1. 가격 조임: {target['price_tightness']:.3f}
        - {'✅ 매우 조여짐' if target['price_tightness'] < 0.3 else '⚠️ 보통' if target['price_tightness'] < 0.5 else '❌ 약함'}
        
        #### 2. 거래량: {target['volume_dryup']:.3f}
        - {'✅ 매도세력 소진' if target['volume_dryup'] < 0.4 else '⚠️ 보통' if target['volume_dryup'] < 0.6 else '❌ 변동 큼'}
        
        #### 3. 레인지: {target['range_ratio']:.3f}
        - {'✅ 매우 좁음' if target['range_ratio'] < 0.4 else '⚠️ 보통' if target['range_ratio'] < 0.6 else '❌ 넓음'}
        
        #### 4. 조용한 양봉 연속:
        - {target['quiet_days']}일 연속 발생
        - ATR: {target['atr']:,.0f}원
        - 기준: 몸통 ≤ ATR × 0.4
        
        #### 5. 저점 유지:
        - 최근 저점: {target['recent_low']:,.0f}원
        - 장기 저점: {target['long_low']:,.0f}원
        - 비율: {(target['recent_low']/target['long_low']):.2%}
        - {'✅ 저점 유지 중' if target['low_hold'] else '❌ 저점 하향'}
        """)
    
    st.info("""
    💡 **이 스캐너는 VCP 판별이 아닌 랭킹입니다**
    - 최종 판단은 차트로 직접 확인
    - 진입가/손절가는 별도 계산기 사용
    """)
