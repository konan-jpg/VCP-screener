import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 0) 기본 설정
# ============================================================
st.set_page_config(page_title="VCP 스캐너 + 진입/손절", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 VCP 스캐너 + 진입/손절 통합")
st.caption("종목코드/종목명 입력 → VCP 체크리스트 + 피벗 + 4가지 진입타점 분석")

# ============================================================
# 1) KRX 종목 리스트 (서버 또는 백업 CSV)
# ============================================================
@st.cache_data(ttl=3600)
def load_krx_listing():
    """KRX 종목 리스트: 서버 실패 시 백업 CSV 사용"""
    try:
        st.info("🔄 KRX 서버 접속 중...")
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        stocks = pd.concat([kospi, kosdaq], ignore_index=True)
        
        # 컬럼 정규화 (환경마다 Symbol/Code 다를 수 있음)
        if 'Symbol' in stocks.columns:
            stocks = stocks.rename(columns={'Symbol': 'Code'})
        elif 'code' in stocks.columns:
            stocks = stocks.rename(columns={'code': 'Code'})
        
        stocks['Code'] = stocks['Code'].astype(str).str.zfill(6)
        result = stocks[['Code', 'Name']].dropna().drop_duplicates()
        
        st.success(f"✅ KRX 서버 접속 성공 ({len(result)}개 종목)")
        return result
        
    except Exception as e:
        st.warning(f"⚠️ KRX 서버 접속 실패: {str(e)}")
        st.info("📂 백업 CSV 사용 중...")
        
        try:
            backup = pd.read_csv('krx_backup.csv')
            if 'Symbol' in backup.columns:
                backup = backup.rename(columns={'Symbol': 'Code'})
            elif 'code' in backup.columns:
                backup = backup.rename(columns={'code': 'Code'})
            
            backup['Code'] = backup['Code'].astype(str).str.zfill(6)
            result = backup[['Code', 'Name']].dropna().drop_duplicates()
            
            st.success(f"✅ 백업 CSV 로딩 완료 ({len(result)}개 종목)")
            return result
            
        except Exception as csv_error:
            st.error(f"❌ 백업 CSV 로딩 실패: {str(csv_error)}")
            return pd.DataFrame(columns=['Code', 'Name'])

def resolve_input(text: str, listing: pd.DataFrame):
    """
    종목명 또는 코드 → [(code, name), ...] 변환
    - 6자리 숫자: 코드로 간주
    - 한글/영문: 종목명 부분일치
    - 여러 후보 → 첫 번째 자동 선택
    """
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    results = []
    
    for line in lines:
        # 6자리 숫자면 코드
        if line.isdigit():
            code = line.zfill(6)
            match = listing[listing['Code'] == code]
            if len(match) > 0:
                results.append((code, match.iloc[0]['Name']))
            else:
                results.append((code, f"미확인({code})"))
        else:
            # 종목명 부분일치
            hits = listing[listing['Name'].str.contains(line, case=False, na=False)]
            if len(hits) == 1:
                results.append((hits.iloc[0]['Code'], hits.iloc[0]['Name']))
            elif len(hits) > 1:
                # 여러 개면 첫 번째 자동 선택
                results.append((hits.iloc[0]['Code'], hits.iloc[0]['Name']))
            else:
                results.append((None, f"미발견({line})"))
    
    # 중복 제거
    seen = set()
    unique = []
    for code, name in results:
        key = (code, name)
        if key not in seen:
            seen.add(key)
            unique.append((code, name))
    
    return unique

# ============================================================
# 2) OHLCV 데이터 + 지표
# ============================================================
@st.cache_data(ttl=3600)
def load_ohlcv(code: str, days=260):
    """주가 데이터 로딩"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start, end)
        
        if df is None or len(df) < 120:
            return None
        return df
    except:
        return None

def add_indicators(df: pd.DataFrame):
    """기술 지표 추가: MA50, VolAvg60, ATR20"""
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['VolAvg60'] = df['Volume'].rolling(60).mean()
    
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR20'] = tr.rolling(20).mean()
    
    return df

# ============================================================
# 3) VCP 스윙/파동 + 체크리스트
# ============================================================
def clean_swings(swings):
    """연속 같은 타입 스윙 정리"""
    if len(swings) < 2:
        return swings
    
    cleaned = [swings[0]]
    for curr in swings[1:]:
        prev = cleaned[-1]
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

def detect_swings(high, low, atr, lookback=60):
    """High/Low 기준 스윙 고점/저점 탐지"""
    if len(high) < lookback:
        return []
    
    h = high.tail(lookback)
    l = low.tail(lookback)
    
    swings = []
    window = 5
    min_size = atr * 1.5
    
    # 고점 탐지
    for i in range(window, len(h) - window):
        v = h.iloc[i]
        is_peak = all(h.iloc[j] < v for j in range(i-window, i+window+1) if j != i)
        if is_peak:
            swings.append({'type': 'high', 'price': float(v), 'date': h.index[i]})
    
    # 저점 탐지
    for i in range(window, len(l) - window):
        v = l.iloc[i]
        is_trough = all(l.iloc[j] > v for j in range(i-window, i+window+1) if j != i)
        if is_trough:
            swings.append({'type': 'low', 'price': float(v), 'date': l.index[i]})
    
    swings.sort(key=lambda x: x['date'])
    swings = clean_swings(swings)
    
    # 최소 크기 필터
    filtered = []
    for s in swings:
        if not filtered:
            filtered.append(s)
        elif abs(s['price'] - filtered[-1]['price']) >= min_size:
            filtered.append(s)
    
    return filtered

def vcp_checklist(df: pd.DataFrame, lookback=60):
    """
    VCP 체크리스트:
    - depth_contraction: 낙폭 감소
    - duration_contraction: 기간 감소
    - highs_tightening: 고점 수렴
    - lows_rising: 저점 상승
    - pivot: 마지막 수축 고점
    """
    atr = df['ATR20'].iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    
    swings = detect_swings(df['High'], df['Low'], atr, lookback)
    
    # high→low 조정 파동 추출
    waves = []
    for i in range(len(swings) - 1):
        if swings[i]['type'] == 'high' and swings[i+1]['type'] == 'low':
            hp = swings[i]['price']
            lp = swings[i+1]['price']
            depth = (hp - lp) / hp
            dur = (swings[i+1]['date'] - swings[i]['date']).days
            
            if depth >= 0.01 and dur >= 2:
                waves.append({
                    'high_price': hp,
                    'low_price': lp,
                    'depth': float(depth),
                    'duration': int(dur),
                })
    
    if len(waves) < 3:
        return {
            'wave_count': len(waves),
            'depth_contraction': False,
            'duration_contraction': False,
            'highs_tightening': False,
            'lows_rising': False,
            'pivot': None,
        }
    
    last3 = waves[-3:]
    pivot = float(last3[-1]['high_price'])
    
    d1, d2, d3 = [w['depth'] for w in last3]
    depth_ok = (d2 <= d1 + 0.01) and (d3 <= d2 + 0.01)
    
    dur1, dur2, dur3 = [w['duration'] for w in last3]
    duration_ok = (dur2 <= dur1 + 3) and (dur3 <= dur2 + 3)
    
    highs = [w['high_price'] for w in last3]
    highs_ok = (max(highs) - min(highs)) <= atr * 1.8
    
    lows = [w['low_price'] for w in last3]
    lows_ok = all(lows[i+1] >= lows[i] - atr*0.5 for i in range(len(lows)-1))
    
    return {
        'wave_count': len(waves),
        'depth_contraction': depth_ok,
        'duration_contraction': duration_ok,
        'highs_tightening': highs_ok,
        'lows_rising': lows_ok,
        'pivot': pivot,
    }

# ============================================================
# 4) VCP 점수 (피벗 근접 우대)
# ============================================================
def vcp_score(df: pd.DataFrame, checklist: dict, short=10, long=60):
    """VCP 점수: 낮을수록 좋음 (피벗 5% 이내 우대)"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']
    atr = df['ATR20'].iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        return None
    
    # 조용함 지표
    std_p_short = close.tail(short).std()
    std_p_long = close.tail(long).std()
    if pd.isna(std_p_long) or std_p_long == 0:
        return None
    price_tight = std_p_short / std_p_long
    
    std_v_short = vol.tail(short).std()
    std_v_long = vol.tail(long).std()
    if pd.isna(std_v_long) or std_v_long == 0:
        return None
    vol_dry = std_v_short / std_v_long
    
    range_pct = (high - low) / close
    range_short = range_pct.tail(short).mean()
    range_long = range_pct.tail(long).mean()
    if pd.isna(range_long) or range_long == 0:
        return None
    range_ratio = range_short / range_long
    
    aux = price_tight * 0.50 + vol_dry * 0.30 + range_ratio * 0.20
    
    # 구조 가중치
    all_ok = all([
        checklist['depth_contraction'],
        checklist['duration_contraction'],
        checklist['highs_tightening'],
        checklist['lows_rising']
    ])
    partial_ok = checklist['depth_contraction'] and checklist['highs_tightening']
    wave_bonus = 0.60 if all_ok else (0.85 if partial_ok else 1.80)
    
    # 피벗 근접 보너스
    cp = float(close.iloc[-1])
    pivot = checklist.get('pivot')
    pivot_dist = None
    pivot_bonus = 1.15
    
    if pivot and pivot > 0:
        pivot_dist = ((pivot - cp) / pivot) * 100
        if pivot_dist <= 0:
            pivot_bonus = 1.20
        elif pivot_dist <= 5:
            pivot_bonus = 0.70 + (pivot_dist / 5) * 0.25
        elif pivot_dist <= 10:
            pivot_bonus = 0.95 + ((pivot_dist - 5) / 5) * 0.20
        else:
            pivot_bonus = 1.15 + min((pivot_dist - 10) / 10, 0.35)
    
    # 추세 보너스
    ma50 = df['MA50'].iloc[-1]
    trend_bonus = 0.90 if (not pd.isna(ma50) and cp >= ma50) else 1.15
    
    score = aux * wave_bonus * pivot_bonus * trend_bonus
    
    return {
        'score': float(score),
        'is_vcp': bool(all_ok),
        'pivot_distance_pct': None if pivot_dist is None else float(pivot_dist),
    }

# ============================================================
# 5) 진입/손절 타점 (Low Cheat는 ATR 버퍼)
# ============================================================
def find_trigger(df, lookback=60):
    """Low Cheat 트리거 바 탐지"""
    x = df.tail(lookback).copy()
    if len(x) < 30:
        return None
    
    atr = x['ATR20']
    vol_avg = x['VolAvg60']
    body = (x['Close'] - x['Open']).abs()
    bullish = x['Close'] > x['Open']
    
    cond = (
        bullish &
        (atr > 0) & atr.notna() &
        (vol_avg > 0) & vol_avg.notna() &
        (body >= 0.6 * atr) &
        (x['Volume'] >= 1.0 * vol_avg)
    )
    
    hits = x[cond]
    return df.loc[hits.index[-1]] if len(hits) > 0 else None

def calc_entries(df, atr_mult=0.3):
    """4가지 진입타점 계산"""
    recent = df.tail(120)
    base_high = float(recent['High'].max())
    base_low = float(recent['Low'].min())
    upper_third = base_low + (base_high - base_low) * 0.66
    
    # 정석 VCP
    vcp_entry = base_high
    vcp_stop = base_high * 0.95
    
    # Cheat
    cheat_zone = recent[recent['High'] >= upper_third]
    cheat_entry = float(cheat_zone['High'].tail(20).max()) if len(cheat_zone) else base_high * 0.98
    cheat_stop = cheat_entry * 0.96
    
    # Low Cheat (ATR 버퍼)
    trig = find_trigger(df, lookback=60)
    if trig is not None and not pd.isna(trig['ATR20']):
        low_entry = float(trig['High'])
        low_stop = max(100.0, float(trig['Low'] - atr_mult * trig['ATR20']))
    else:
        low_entry = float(recent['High'].tail(10).max())
        atr20 = recent['ATR20'].iloc[-1]
        buffer = float(atr_mult * atr20) if not pd.isna(atr20) else 0.0
        low_stop = max(100.0, float(recent['Low'].tail(10).min() - buffer))
    
    # Pullback
    pull_entry = base_high
    pull_stop = base_high * 0.97
    
    entries = [
        ('정석 VCP', vcp_entry, vcp_stop),
        ('Cheat', cheat_entry, cheat_stop),
        ('Low Cheat', low_entry, low_stop),
        ('Pullback', pull_entry, pull_stop),
    ]
    
    rows = []
    for name, entry, stop in entries:
        r = entry - stop
        risk_pct = (entry - stop) / entry * 100 if entry > 0 else 0.0
        rows.append({
            '타점': name,
            '진입가': entry,
            '손절가': stop,
            'R(원)': r,
            '손절폭(%)': risk_pct,
        })
    
    return pd.DataFrame(rows)

# ============================================================
# 6) 차트 (캔들 + 50MA + 피벗)
# ============================================================
def plot_chart(df, name, code, pivot=None):
    """차트 (피벗 라인만 표시, 진입/손절 라인 제거)"""
    d = df.tail(120)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d['Open'],
        high=d['High'],
        low=d['Low'],
        close=d['Close'],
        name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=d.index,
        y=d['MA50'],
        name='50MA',
        line=dict(color='blue', dash='dot', width=1)
    ), row=1, col=1)
    
    if pivot:
        fig.add_trace(go.Scatter(
            x=[d.index[0], d.index[-1]],
            y=[pivot, pivot],
            name=f'Pivot ({pivot:,.0f})',
            line=dict(color='red', dash='dash', width=2)
        ), row=1, col=1)
    
    colors = ['red' if r.Open > r.Close else 'green' for r in d.itertuples()]
    fig.add_trace(go.Bar(
        x=d.index,
        y=d['Volume'],
        marker_color=colors,
        name='Volume'
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        title=f'{name} ({code})',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# ============================================================
# 7) UI
# ============================================================
listing = load_krx_listing()

with st.sidebar:
    st.header('⚙️ 설정')
    
    st.markdown('### 📥 입력')
    input_text = st.text_area(
        '종목코드 또는 종목명 (줄바꿈 입력)',
        value='005930\n000660\n삼성바이오로직스',
        height=160
    )
    
    st.markdown('### 🔬 파라미터')
    short_period = st.slider('단기(조용함)', 5, 20, 10, 1)
    long_period = st.slider('장기(기준)', 40, 120, 60, 5)
    atr_mult = st.slider('Low Cheat ATR 버퍼', 0.1, 1.0, 0.3, 0.1)
    
    st.divider()
    run = st.button('🚀 분석 실행', type='primary', use_container_width=True)

if 'results' not in st.session_state:
    st.session_state.results = []
if 'selected' not in st.session_state:
    st.session_state.selected = 0

# ============================================================
# 8) 실행
# ============================================================
if run:
    resolved = resolve_input(input_text, listing)
    
    rows = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    for idx, (code, name) in enumerate(resolved):
        progress_bar.progress((idx + 1) / len(resolved))
        status.text(f'{idx+1}/{len(resolved)} - {name}')
        
        if code is None:
            rows.append({
                'Code': '',
                'Name': name,
                'Error': 'NOT_FOUND'
            })
            continue
        
        df = load_ohlcv(code)
        if df is None:
            rows.append({
                'Code': code,
                'Name': name,
                'Error': 'NO_DATA'
            })
            continue
        
        df = add_indicators(df)
        checklist = vcp_checklist(df, lookback=60)
        
        if checklist is None:
            rows.append({
                'Code': code,
                'Name': name,
                'Error': 'INDICATOR_FAIL'
            })
            continue
        
        score_pack = vcp_score(df, checklist, short=short_period, long=long_period)
        
        if score_pack is None:
            rows.append({
                'Code': code,
                'Name': name,
                'Error': 'SCORE_FAIL'
            })
            continue
        
        cp = float(df['Close'].iloc[-1])
        
        rows.append({
            'Code': code,
            'Name': name,
            'Error': None,
            '현재가': cp,
            '점수': score_pack['score'],
            '완전VCP': '✅' if score_pack['is_vcp'] else '❌',
            '파동수': checklist['wave_count'],
            '저점상승': '✅' if checklist['lows_rising'] else '❌',
            '깊이수축': '✅' if checklist['depth_contraction'] else '❌',
            '고점수렴': '✅' if checklist['highs_tightening'] else '❌',
            '기간수축': '✅' if checklist['duration_contraction'] else '❌',
            '피벗': checklist.get('pivot'),
            '피벗거리%': score_pack.get('pivot_distance_pct'),
            '_df': df,
            '_checklist': checklist,
        })
    
    progress_bar.empty()
    status.empty()
    
    out = pd.DataFrame(rows)
    ok = out[out['Error'].isna()].copy()
    
    if len(ok) == 0:
        st.warning('⚠️ 분석 가능한 종목이 없습니다.')
        st.session_state.results = []
        st.stop()
    
    ok = ok.sort_values('점수').reset_index(drop=True)
    ok.insert(0, '순위', range(1, len(ok) + 1))
    
    st.session_state.results = ok.to_dict('records')
    st.session_state.selected = 0
    
    st.success(f'✅ {len(ok)}개 분석 완료')

results = st.session_state.results

if not results:
    st.info('👈 종목 입력 후 분석 실행을 누르세요.')
    st.stop()

# ============================================================
# 9) 테이블 (클릭 연동)
# ============================================================
table_rows = []
for r in results:
    pivot_str = 'N/A'
    if r.get('피벗'):
        pivot_str = f"{r['피벗']:,.0f}"
    
    dist_str = 'N/A'
    if r.get('피벗거리%') is not None:
        dist_str = f"{r['피벗거리%']:.1f}%"
    
    table_rows.append({
        '순위': r['순위'],
        '종목': r['Name'],
        '코드': r['Code'],
        '현재가': f"{r['현재가']:,.0f}",
        '점수': f"{r['점수']:.3f}",
        '완전VCP': r['완전VCP'],
        '피벗': pivot_str,
        '피벗거리': dist_str,
        '저점↑': r['저점상승'],
        '깊이↓': r['깊이수축'],
        '고점→': r['고점수렴'],
        '기간↓': r['기간수축'],
        '파동': r['파동수'],
    })

summary_df = pd.DataFrame(table_rows)

event = st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    on_select='rerun',
    selection_mode='single-row'
)

if event.selection.rows:
    st.session_state.selected = event.selection.rows[0]

# ============================================================
# 10) 상세 (차트 + 체크리스트 + 타점)
# ============================================================
idx = max(0, min(st.session_state.selected, len(results) - 1))
target = results[idx]

df = target['_df']
checklist = target['_checklist']
entries_df = calc_entries(df, atr_mult=atr_mult)

st.divider()
st.subheader(f"📌 {target['Name']} ({target['Code']})")

m1, m2, m3, m4 = st.columns(4)
m1.metric('현재가', f"{target['현재가']:,.0f}")
m2.metric('점수', f"{target['점수']:.3f}")
m3.metric('완전 VCP', target['완전VCP'])

pivot_val = target.get('피벗')
m4.metric('피벗', 'N/A' if not pivot_val else f"{pivot_val:,.0f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric('저점 상승', target['저점상승'])
c2.metric('깊이 수축', target['깊이수축'])
c3.metric('고점 수렴', target['고점수렴'])
c4.metric('기간 수축', target['기간수축'])

st.plotly_chart(
    plot_chart(df, target['Name'], target['Code'], pivot=pivot_val),
    use_container_width=True
)

st.markdown('### 🎯 진입/손절 타점')
disp = entries_df.copy()
disp['진입가'] = disp['진입가'].map(lambda x: f'{x:,.0f}')
disp['손절가'] = disp['손절가'].map(lambda x: f'{x:,.0f}')
disp['R(원)'] = disp['R(원)'].map(lambda x: f'{x:,.0f}')
disp['손절폭(%)'] = disp['손절폭(%)'].map(lambda x: f'{x:.1f}%')

st.dataframe(
    disp[['타점', '진입가', '손절가', 'R(원)', '손절폭(%)']],
    use_container_width=True,
    hide_index=True
)
