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
st.set_page_config(page_title="VCP Scanner v4 Final", layout="wide")

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
    """KRX 종목 리스트 로딩 (백업 CSV 포함)"""
    try:
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
        
        st.success("✅ KRX 서버 접속 성공")
        return stocks[['Code', 'Name', 'Market', 'Marcap_billion']]
        
    except Exception as e:
        st.warning(f"⚠️ KRX 서버 접속 실패: {str(e)}")
        st.info("📂 백업 CSV 사용 중...")
        
        try:
            backup_df = pd.read_csv('krx_backup.csv')
            backup_df = backup_df[backup_df['Marcap'] >= 200_000_000_000]
            backup_df = backup_df.sort_values('Marcap', ascending=False)
            backup_df['Marcap_billion'] = backup_df['Marcap'] / 100_000_000
            
            st.success(f"✅ 백업 CSV 로딩 완료 ({len(backup_df)}개)")
            return backup_df[['Code', 'Name', 'Market', 'Marcap_billion']]
            
        except Exception as csv_error:
            st.error(f"❌ 백업 CSV 로딩 실패: {str(csv_error)}")
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
# 3. VCP 파동 구조 인식 함수
# -----------------------------------------------------------
def clean_zigzag_swings(swings):
    """연속된 같은 타입의 스윙 정리"""
    if len(swings) < 2:
        return swings
    
    cleaned = [swings[0]]
    
    for i in range(1, len(swings)):
        prev = cleaned[-1]
        curr = swings[i]
        
        if prev['type'] == curr['type']:
            if prev['type'] == 'high':
                if curr['price'] > prev['price']:
                    cleaned[-1] = curr
            else:
                if curr['price'] < prev['price']:
                    cleaned[-1] = curr
        else:
            cleaned.append(curr)
    
    return cleaned

def detect_swings_hl(high, low, close, atr, lookback=60):
    """High/Low 기준 스윙 고점·저점 추출"""
    if len(high) < lookback:
        return []
    
    high_series = high.tail(lookback)
    low_series = low.tail(lookback)
    
    swings = []
    window = 5
    min_swing_size = atr * 1.5
    
    for i in range(window, len(high_series) - window):
        local_high = high_series.iloc[i]
        is_peak = True
        
        for j in range(i - window, i + window + 1):
            if j != i and high_series.iloc[j] >= local_high:
                is_peak = False
                break
        
        if is_peak:
            swings.append({
                'type': 'high',
                'price': local_high,
                'date': high_series.index[i],
                'idx': i
            })
    
    for i in range(window, len(low_series) - window):
        local_low = low_series.iloc[i]
        is_trough = True
        
        for j in range(i - window, i + window + 1):
            if j != i and low_series.iloc[j] <= local_low:
                is_trough = False
                break
        
        if is_trough:
            swings.append({
                'type': 'low',
                'price': local_low,
                'date': low_series.index[i],
                'idx': i
            })
    
    swings.sort(key=lambda x: x['date'])
    swings = clean_zigzag_swings(swings)
    
    filtered_swings = []
    for i in range(len(swings)):
        if i == 0:
            filtered_swings.append(swings[i])
            continue
        
        prev_price = filtered_swings[-1]['price']
        curr_price = swings[i]['price']
        move_size = abs(curr_price - prev_price)
        
        if move_size >= min_swing_size:
            filtered_swings.append(swings[i])
    
    return filtered_swings

def validate_vcp_structure(swings, atr):
    """VCP 구조 검증"""
    if len(swings) < 6:
        return {
            'is_vcp': False,
            'wave_bonus': 1.8,
            'depth_contraction': False,
            'duration_contraction': False,
            'highs_tightening': False,
            'lows_rising': False,
            'waves': []
        }
    
    correction_waves = []
    for i in range(len(swings) - 1):
        if swings[i]['type'] == 'high' and swings[i+1]['type'] == 'low':
            high_price = swings[i]['price']
            low_price = swings[i+1]['price']
            
            depth = (high_price - low_price) / high_price
            duration = (swings[i+1]['date'] - swings[i]['date']).days
            
            if depth >= 0.01 and duration >= 2:
                correction_waves.append({
                    'high_price': high_price,
                    'low_price': low_price,
                    'high_date': swings[i]['date'],
                    'low_date': swings[i+1]['date'],
                    'depth': depth,
                    'duration': duration
                })
    
    if len(correction_waves) < 3:
        return {
            'is_vcp': False,
            'wave_bonus': 1.8,
            'depth_contraction': False,
            'duration_contraction': False,
            'highs_tightening': False,
            'lows_rising': False,
            'waves': correction_waves
        }
    
    last_3_waves = correction_waves[-3:]
    
    d1 = last_3_waves[0]['depth']
    d2 = last_3_waves[1]['depth']
    d3 = last_3_waves[2]['depth']
    
    dur1 = last_3_waves[0]['duration']
    dur2 = last_3_waves[1]['duration']
    dur3 = last_3_waves[2]['duration']
    
    depth_tolerance = 0.01
    depth_contraction = (d2 <= d1 + depth_tolerance) and (d3 <= d2 + depth_tolerance)
    
    duration_tolerance = 3
    duration_contraction = (dur2 <= dur1 + duration_tolerance) and (dur3 <= dur2 + duration_tolerance)
    
    recent_highs = [w['high_price'] for w in last_3_waves]
    high_range = max(recent_highs) - min(recent_highs)
    highs_tightening = high_range <= atr * 1.8
    
    recent_lows = [w['low_price'] for w in last_3_waves]
    low_tolerance = atr * 0.5
    lows_rising = all(
        recent_lows[i+1] >= recent_lows[i] - low_tolerance 
        for i in range(len(recent_lows) - 1)
    )
    
    is_vcp = depth_contraction and duration_contraction and highs_tightening and lows_rising
    
    if is_vcp:
        wave_bonus = 0.60
    elif depth_contraction and highs_tightening:
        wave_bonus = 0.85
    else:
        wave_bonus = 1.8
    
    return {
        'is_vcp': is_vcp,
        'wave_bonus': wave_bonus,
        'depth_contraction': depth_contraction,
        'duration_contraction': duration_contraction,
        'highs_tightening': highs_tightening,
        'lows_rising': lows_rising,
        'waves': correction_waves
    }

# -----------------------------------------------------------
# 4. VCP Scanner v4 Final
# -----------------------------------------------------------
def vcp_tightness_scanner(df, short_period=10, long_period=60, atr_period=20):
    """VCP Scanner v4 Final"""
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
    
    recent5_vol = volume.tail(5).mean()
    recent5_range = ((high.tail(5) - low.tail(5)) / close.tail(5)).mean()
    
    if recent5_vol == 0 or recent5_range < 0.005:
        return None
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        return None
    
    swings = detect_swings_hl(high, low, close, atr, lookback=60)
    vcp_result = validate_vcp_structure(swings, atr)
    
    std_price_short = close.tail(short_period).std()
    std_price_long = close.tail(long_period).std()
    
    if std_price_long == 0 or pd.isna(std_price_long):
        return None
    
    price_tightness = std_price_short / std_price_long
    
    std_vol_short = volume.tail(short_period).std()
    std_vol_long = volume.tail(long_period).std()
    
    if std_vol_long == 0 or pd.isna(std_vol_long):
        return None
    
    volume_dryup = std_vol_short / std_vol_long
    
    range_pct = (high - low) / close
    range_short = range_pct.tail(short_period).mean()
    range_long = range_pct.tail(long_period).mean()
    
    if range_long == 0 or pd.isna(range_long):
        return None
    
    range_ratio = range_short / range_long
    
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
    
    recent_low = low.tail(short_period).min()
    long_low = low.tail(long_period).min()
    
    low_hold = recent_low >= long_low * 1.01
    low_hold_bonus = 0.90 if low_hold else 1.0
    
    auxiliary_score = (
        price_tightness * 0.50 +
        volume_dryup * 0.30 +
        range_ratio * 0.20
    )
    
    structural_score = auxiliary_score * vcp_result['wave_bonus']
    final_score = structural_score * quiet_bonus * low_hold_bonus
    
    return {
        "score": final_score,
        "auxiliary_score": auxiliary_score,
        "is_vcp": vcp_result['is_vcp'],
        "wave_bonus": vcp_result['wave_bonus'],
        "depth_contraction": vcp_result['depth_contraction'],
        "duration_contraction": vcp_result['duration_contraction'],
        "highs_tightening": vcp_result['highs_tightening'],
        "lows_rising": vcp_result['lows_rising'],
        "wave_count": len(vcp_result['waves']),
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
# 5. 차트
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
        vcp_icon = "✅ VCP" if result.get('is_vcp') else "⚠️" if result.get('wave_bonus') < 1.5 else "❌"
        structure = []
        if result.get('depth_contraction'): structure.append("깊이↓")
        if result.get('highs_tightening'): structure.append("고점→")
        if result.get('lows_rising'): structure.append("저점↑")
        
        title += f" | {vcp_icon} | 점수: {result['score']:.3f} | {' '.join(structure)}"
    
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
st.title("🔍 VCP Scanner v4 Final")
st.markdown("""
**High/Low 기반 파동 구조 인식 스캐너**

✅ **구조 검증**: 깊이 수축 + 고점 압력 감소 + 저점 지지 상승  
✅ **클릭으로 차트 확인**: 테이블에서 행을 클릭하면 바로 차트가 변경됩니다
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
    top_n = st.slider("상위 N개", 10, 100, 30, 5)
    
    st.divider()
    
    if st.button("🚀 스캔", type="primary", use_container_width=True):
        st.session_state['run'] = True
        st.session_state['results'] = []

if 'results' not in st.session_state:
    st.session_state['results'] = []

# -----------------------------------------------------------
# 7. 스캔 실행
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
            
            vcp_count = sum([1 for r in ranking.to_dict('records') if r.get('is_vcp')])
            partial_count = sum([1 for r in ranking.to_dict('records') if not r.get('is_vcp') and r.get('wave_bonus') < 1.5])
            st.success(f"✅ {len(ranking)}개 발견! (완전 VCP: {vcp_count}개 / 부분 통과: {partial_count}개)")

# -----------------------------------------------------------
# 8. 결과 (테이블 클릭 → 차트 변경)
# -----------------------------------------------------------
results = st.session_state['results']

if not results:
    st.info("👈 설정 후 스캔")
else:
    vcp_count = sum([1 for r in results if r.get('is_vcp')])
    partial_count = sum([1 for r in results if not r.get('is_vcp') and r.get('wave_bonus') < 1.5])
    
    st.success(f"🎯 상위 {len(results)}개 | 완전 VCP: {vcp_count}개 | 부분 통과: {partial_count}개")
    
    st.markdown("### 📋 전체 랭킹 (행을 클릭하면 아래 차트가 변경됩니다)")
    
    # 테이블용 DataFrame 생성
    summary_df = pd.DataFrame([{
        '순위': idx + 1,
        '종목': r['Name'],
        'VCP': '✅' if r.get('is_vcp') else '⚠️' if r.get('wave_bonus') < 1.5 else '❌',
        '시총(억)': f"{r['Marcap']:,.0f}",
        '현재가': f"{r['current_price']:,.0f}",
        '점수': f"{r['score']:.3f}",
        '깊이': '✅' if r.get('depth_contraction') else '❌',
        '고점': '✅' if r.get('highs_tightening') else '❌',
        '저점': '✅' if r.get('lows_rising') else '❌',
        '파동': r.get('wave_count', 0)
    } for idx, r in enumerate(results)])
    
    # st.data_editor로 행 선택 가능
    edited_df = st.data_editor(
        summary_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        key="ranking_table",
        disabled=True
    )
    
    st.divider()
    
    # ✅ 수정된 부분: 올바른 API 호출
    selected_rows = st.session_state.get("ranking_table", {}).get("selected_rows", [])
    
    if selected_rows:
        selected_idx = selected_rows[0]
        target = results[selected_idx]
        
        st.subheader(f"📊 {target['Name']} 상세 분석")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("순위", f"{selected_idx + 1}")
        col2.metric("완전 VCP", "✅" if target.get('is_vcp') else "❌")
        col3.metric("점수", f"{target['score']:.3f}")
        col4.metric("파동 배수", f"{target['wave_bonus']:.2f}x")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("깊이 수축", "✅" if target.get('depth_contraction') else "❌")
        col6.metric("고점 압력↓", "✅" if target.get('highs_tightening') else "❌")
        col7.metric("저점 지지↑", "✅" if target.get('lows_rising') else "❌")
        col8.metric("파동 수", target.get('wave_count', 0))
        
        fig = plot_chart(target['df'], target['Name'], target['Code'], target)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔬 상세 지표"):
            detail_df = pd.DataFrame([{
                '지표': '보조 점수',
                '값': f"{target['auxiliary_score']:.3f}"
            }, {
                '지표': 'Price Tightness',
                '값': f"{target['price_tightness']:.3f}"
            }, {
                '지표': 'Volume Dry-up',
                '값': f"{target['volume_dryup']:.3f}"
            }, {
                '지표': 'Range Ratio',
                '값': f"{target['range_ratio']:.3f}"
            }, {
                '지표': '조용한 양봉',
                '값': f"{target['quiet_days']}일"
            }, {
                '지표': '저점 유지',
                '값': '✅' if target.get('low_hold') else '❌'
            }, {
                '지표': 'ATR',
                '값': f"{target['atr']:,.0f}"
            }])
            
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 위 테이블에서 종목을 클릭하면 차트가 표시됩니다")

