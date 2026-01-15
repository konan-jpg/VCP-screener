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
    .warning-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 2. 종목 리스트 로딩 (백업 CSV 포함)
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_krx_stocks():
    """
    KRX 종목 리스트 로딩 (백업 CSV 포함)
    1. KRX 서버 접속 시도
    2. 실패 시 백업 CSV 사용
    3. 성공 시 캐시 업데이트
    """
    try:
        # KRX 서버 접속 시도
        st.info("🔄 KRX 서버 접속 중...")
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq])
        
        stocks = stocks[~stocks['Name'].str.contains('우')]
        stocks = stocks[~stocks['Name'].str.contains('스팩')]
        
        if 'Marcap' in stocks.columns:
            stocks = stocks[stocks['Marcap'] >= 200_000_000_000]
            stocks = stocks.sort_values('Marcap', ascending=False)
            stocks['Marcap_billion'] = stocks['Marcap'] / 100_000_000
        
        st.success("✅ KRX 서버 접속 성공 - 최신 데이터 사용")
        return stocks[['Code', 'Name', 'Market', 'Marcap_billion']]
        
    except Exception as e:
        # KRX 접속 실패 시 백업 CSV 사용
        st.warning(f"⚠️ KRX 서버 접속 실패: {str(e)}")
        st.info("📂 백업 CSV 파일 사용 중...")
        
        try:
            # GitHub에 업로드된 백업 파일 읽기
            backup_df = pd.read_csv('krx_backup.csv')
            
            # 시총 2,000억 이상 필터
            backup_df = backup_df[backup_df['Marcap'] >= 200_000_000_000]
            backup_df = backup_df.sort_values('Marcap', ascending=False)
            backup_df['Marcap_billion'] = backup_df['Marcap'] / 100_000_000
            
            st.success(f"✅ 백업 CSV 로딩 완료 ({len(backup_df)}개 종목)")
            return backup_df[['Code', 'Name', 'Market', 'Marcap_billion']]
            
        except Exception as csv_error:
            st.error(f"❌ 백업 CSV 로딩 실패: {str(csv_error)}")
            st.error("krx_backup.csv 파일이 GitHub 저장소에 있는지 확인하세요")
            return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_data(code, days=200):
    """주식 데이터 (백업 포함)"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 0 else None
    except Exception as e:
        st.warning(f"⚠️ {code} 데이터 로딩 실패: {str(e)}")
        return None

# -----------------------------------------------------------
# 3. VCP Tightness Scanner v3
# -----------------------------------------------------------
def vcp_tightness_scanner(df, short_period=10, long_period=60, atr_period=20):
    """VCP Tightness Scanner v3"""
    if df is None or len(df) < long_period + atr_period:
        return None
    
    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    current_price = close.iloc[-1]
    if current_price < 10_000:
        return None
    
    # 1. Price Tightness
    std_price_short = close.tail(short_period).std()
    std_price_long = close.tail(long_period).std()
    
    if std_price_long == 0 or pd.isna(std_price_long):
        return None
    
    price_tightness = std_price_short / std_price_long
    
    # 2. Volume Dry-up
    std_vol_short = volume.tail(short_period).std()
    std_vol_long = volume.tail(long_period).std()
    
    if std_vol_long == 0 or pd.isna(std_vol_long):
        return None
    
    volume_dryup = std_vol_short / std_vol_long
    
    # 3. Range Contraction
    range_pct = (high - low) / close
    range_short = range_pct.tail(short_period).mean()
    range_long = range_pct.tail(long_period).mean()
    
    if range_long == 0 or pd.isna(range_long):
        return None
    
    range_ratio = range_short / range_long
    
    # 4. ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        return None
    
    # 5. 조용한 양봉 연속성
    quiet_days = 0
    for i in range(1, 4):
        if len(close) < i:
            break
        
        day_close = close.iloc[-i]
        day_open = open_.iloc[-i]
        body = abs(day_close - day_open)
        
        if day_close > day_open and body <= atr * 0.40:
            quiet_days += 1
    
    quiet_bonus = 1.0 - min(quiet_days * 0.05, 0.15)
    
    # 6. 저점 유지력
    recent_low = low.tail(short_period).min()
    long_low = low.tail(long_period).min()
    
    low_hold = recent_low >= long_low * 1.01
    low_hold_bonus = 0.90 if low_hold else 1.0
    
    # 7. 점수 계산
    base_score = (
        price_tightness * 0.50 +
        volume_dryup * 0.30 +
        range_ratio * 0.20
    )
    
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
# 4. 차트
# -----------------------------------------------------------
def plot_chart(df, name, code, result):
    """차트"""
    df_chart = df.tail(120)
    
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
    
    ma50 = df_chart['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=ma50,
        line=dict(color='blue', width=1, dash='dot'),
        name='50MA'
    ), row=1, col=1)
    
    colors = ['red' if r.Open > r.Close else 'green' for r in df_chart.itertuples()]
    fig.add_trace(go.Bar(
        x=df_chart.index,
        y=df_chart['Volume'],
        marker_color=colors
    ), row=2, col=1)
    
    title = f"{name} ({code})"
    if result:
        title += f" | 점수: {result['score']:.3f} | 양봉: {result['quiet_days']}일"
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# -----------------------------------------------------------
# 5. UI
# -----------------------------------------------------------
st.title("🔍 VCP Tightness Scanner v3")
st.markdown("""
**KRX 접속 실패 시 백업 CSV 자동 사용**

✅ **하드 필터**: 시총 2,000억+ / 현재가 10,000원+  
✅ **보너스**: 저점유지 10% / 조용한양봉 최대 15%  
✅ **백업**: KRX 접속 실패 시 자동으로 백업 CSV 사용
""")

with st.sidebar:
    st.header("⚙️ 설정")
    
    st.markdown("### 📊 스캔")
    scan_count = st.selectbox("시총 상위", [100, 300, 500, 1000], index=1)
    
    st.divider()
    
    st.markdown("### 🔬 파라미터")
    short_period = st.slider("단기", 5, 20, 10, 1)
    long_period = st.slider("장기", 40, 120, 60, 5)
    atr_period = st.slider("ATR", 10, 30, 20, 5)
    
    st.divider()
    
    st.markdown("### 🎯 결과")
    top_n = st.slider("상위", 10, 100, 30, 5)
    
    st.divider()
    
    if st.button("🚀 스캔", type="primary", use_container_width=True):
        st.session_state['run'] = True
        st.session_state['results'] = []

if 'results' not in st.session_state:
    st.session_state['results'] = []

# -----------------------------------------------------------
# 6. 스캔 실행
# -----------------------------------------------------------
if st.session_state.get('run'):
    stocks = get_krx_stocks()
    
    if stocks.empty:
        st.error("❌ 종목 로딩 실패")
        st.session_state['run'] = False
    else:
        stocks_to_scan = stocks.head(scan_count)
        
        st.info(f"📊 {len(stocks_to_scan)}개 스캔 중...")
        
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
            st.warning("조건 맞는 종목 없음")
            st.session_state['run'] = False
        else:
            ranking = pd.DataFrame(results).sort_values('score').head(top_n)
            st.session_state['results'] = ranking.to_dict('records')
            st.session_state['run'] = False
            
            st.success(f"✅ {len(ranking)}개 발견!")

# -----------------------------------------------------------
# 7. 결과
# -----------------------------------------------------------
results = st.session_state['results']

if not results:
    st.info("👈 설정 후 스캔")
    
    with st.expander("💡 백업 CSV 사용법"):
        st.markdown("""
        ### krx_backup.csv 만들기
        
        **필수 컬럼:**
        ```csv
        Code,Name,Market,Marcap
        005930,삼성전자,KOSPI,500000000000000
        000660,SK하이닉스,KOSPI,100000000000000
        ```
        
        **준비 방법:**
        1. 엑셀이나 구글 시트에서 작성
        2. CSV로 저장
        3. GitHub 저장소 루트에 업로드
        
        **작동 방식:**
        1. KRX 서버 접속 시도
        2. 성공 → 최신 데이터 사용 & 캐시 저장
        3. 실패 → 백업 CSV 사용
        4. 캐시 유지로 최신 상태 보존
        """)
else:
    st.success(f"🎯 상위 {len(results)}개")
    
    with st.expander("📋 랭킹", expanded=True):
        summary = pd.DataFrame([{
            '순위': idx + 1,
            '종목': r['Name'],
            '시총(억)': f"{r['Marcap']:,.0f}",
            '현재가': f"{r['current_price']:,.0f}",
            '점수': f"{r['score']:.3f}",
            '양봉': f"{r['quiet_days']}일",
            '저점': '✅' if r['low_hold'] else '❌'
        } for idx, r in enumerate(results)])
        
        st.dataframe(summary, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("📊 상세")
    
    selected = st.selectbox(
        "종목",
        [f"{idx+1}. {r['Name']} - {r['score']:.3f}" 
         for idx, r in enumerate(results)]
    )
    
    idx = int(selected.split('.')[0]) - 1
    target = results[idx]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("순위", f"{idx + 1}")
    col2.metric("점수", f"{target['score']:.3f}")
    col3.metric("기본", f"{target['base_score']:.3f}")
    col4.metric("양봉", f"{target['quiet_days']}일")
    col5.metric("저점", "✅" if target['low_hold'] else "❌")
    
    fig = plot_chart(target['df'], target['Name'], target['Code'], target)
    st.plotly_chart(fig, use_container_width=True)
