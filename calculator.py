import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="VCP Auto Calculator", layout="wide")
st.title("🎯 VCP 자동 계산기")

st.markdown("""
**종목 코드만 입력하면 Pivot · Stop · Target 자동 계산**

- ✅ Pivot 자동 계산 (최근 고점)
- ✅ Stop 자동 계산 (Pivot 기준 5~7%)
- ✅ 2R, 3R 목표가 자동 표시
- ✅ 포지션 사이징 계산
""")

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(code):
    """주식 데이터 로딩"""
    end = datetime.now()
    start = end - timedelta(days=200)
    
    try:
        df = fdr.DataReader(code, start, end)
        if df is not None and len(df) > 0:
            return df
        return None
    except:
        return None

# -------------------------------------------------
# Pivot & Stop 자동 계산
# -------------------------------------------------
def calculate_pivot_stop(df, pivot_period=60, stop_pct=5.0):
    """
    Pivot & Stop 자동 계산
    
    Pivot: 최근 60일 최고가
    Stop: Pivot 기준 5% 하락
    """
    if df is None or len(df) < pivot_period:
        return None, None, "데이터 부족"
    
    # Pivot = 최근 N일 최고가
    recent_df = df.tail(pivot_period)
    pivot = recent_df['High'].max()
    
    # Pivot이 나온 날짜
    pivot_date = recent_df[recent_df['High'] == pivot].index[-1]
    days_ago = (df.index[-1] - pivot_date).days
    
    # Stop = Pivot에서 N% 하락
    stop = pivot * (1 - stop_pct / 100)
    
    return pivot, stop, f"Pivot: {days_ago}일 전 고점"

# -------------------------------------------------
# 포지션 사이징 계산
# -------------------------------------------------
def calculate_position_sizing(account, risk_pct, pivot, stop):
    """포지션 사이징"""
    if pivot <= 0 or stop <= 0 or pivot <= stop:
        return 0, 0, 0, 0
    
    risk_amount = account * (risk_pct / 100)
    loss_per_share = pivot - stop
    
    if loss_per_share <= 0:
        return 0, 0, 0, 0
    
    qty = int(risk_amount / loss_per_share)
    total = qty * pivot
    position_pct = (total / account) * 100
    
    return qty, total, position_pct, loss_per_share

# -------------------------------------------------
# UI - 입력부
# -------------------------------------------------
col_input, col_chart = st.columns([1, 3])

with col_input:
    st.subheader("📥 입력")
    
    code = st.text_input(
        "종목 코드",
        placeholder="예: 005930",
        help="6자리 종목 코드"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ 설정")
    
    pivot_period = st.slider(
        "Pivot 기간 (일)",
        30, 120, 60, 5,
        help="최근 N일 중 최고가를 Pivot으로"
    )
    
    stop_pct = st.slider(
        "손절폭 (%)",
        3.0, 10.0, 5.0, 0.5,
        help="Pivot 대비 하락 %"
    )
    
    st.divider()
    
    st.markdown("### 💰 자금 관리")
    
    account = st.number_input(
        "총 자산 (원)",
        value=50_000_000,
        step=1_000_000,
        format="%d"
    )
    
    risk_pct = st.slider(
        "계좌 리스크 (%)",
        0.5, 2.5, 1.0, 0.1,
        help="한 번 매매 시 전체 자산 중 리스크 비율"
    )
    
    st.divider()
    
    st.markdown("### 🎯 목표가")
    show_2r = st.checkbox("2R 표시", value=True)
    show_3r = st.checkbox("3R 표시", value=True)

# -------------------------------------------------
# 차트 & 계산 결과
# -------------------------------------------------
with col_chart:
    if not code:
        st.info("👈 종목 코드를 입력하세요")
        
        with st.expander("💡 사용법"):
            st.markdown("""
            ### 자동 계산 방식
            
            **Pivot 계산:**
            - 최근 60일(기본) 중 **최고가**
            - 슬라이더로 기간 조정 가능
            - 이것이 진입 목표가
            
            **Stop 계산:**
            - Pivot에서 5%(기본) 하락
            - 슬라이더로 손절폭 조정
            - 이 가격 이탈 시 무조건 청산
            
            **포지션 사이징:**
            - 계좌 리스크: 1% (기본)
            - 한 번 손절 시 총 자산의 1% 손실
            - 이에 맞는 수량 자동 계산
            
            **R 배수:**
            - 1R = Pivot - Stop
            - 2R = Pivot + (1R × 2)
            - 3R = Pivot + (1R × 3)
            """)
    else:
        df = load_data(code)
        
        if df is None:
            st.error("❌ 데이터 로딩 실패")
            st.info("종목 코드를 확인하세요")
        else:
            # Pivot & Stop 계산
            pivot, stop, info_msg = calculate_pivot_stop(df, pivot_period, stop_pct)
            
            if pivot is None:
                st.error("계산 실패")
            else:
                current_price = df['Close'].iloc[-1]
                
                # 주요 지표 표시
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    "현재가",
                    f"{current_price:,.0f}원"
                )
                
                col2.metric(
                    "🎯 Pivot (진입가)",
                    f"{pivot:,.0f}원",
                    f"+{((pivot - current_price) / current_price * 100):.1f}%"
                )
                
                col3.metric(
                    "🛑 Stop (손절가)",
                    f"{stop:,.0f}원",
                    f"-{stop_pct}%"
                )
                
                r_value = pivot - stop
                col4.metric(
                    "1R",
                    f"{r_value:,.0f}원"
                )
                
                st.caption(info_msg)
                
                # 포지션 사이징
                qty, total, pos_pct, loss_per_share = calculate_position_sizing(
                    account, risk_pct, pivot, stop
                )
                
                st.divider()
                
                st.markdown("### 💼 포지션 사이징")
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "매수 수량",
                    f"{qty:,}주",
                    help=f"주당 손실: {loss_per_share:,.0f}원"
                )
                
                col2.metric(
                    "투입 금액",
                    f"{total:,.0f}원",
                    f"비중 {pos_pct:.1f}%"
                )
                
                max_loss = qty * loss_per_share
                col3.metric(
                    "최대 손실",
                    f"{max_loss:,.0f}원",
                    f"계좌의 {risk_pct}%"
                )
                
                if pos_pct > 20:
                    st.error(f"⚠️ 비중 {pos_pct:.1f}%는 과도합니다!")
                elif pos_pct > 15:
                    st.warning(f"⚠️ 비중 {pos_pct:.1f}%는 다소 높습니다")
                
                # 차트
                st.divider()
                st.markdown("### 📈 차트")
                
                fig = go.Figure()
                
                # 캔들
                df_chart = df.tail(120)
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart['Open'],
                    high=df_chart['High'],
                    low=df_chart['Low'],
                    close=df_chart['Close'],
                    name="Price"
                ))
                
                # 50일선
                ma50 = df_chart['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(
                    x=df_chart.index,
                    y=ma50,
                    line=dict(color='blue', width=1, dash='dot'),
                    name='50MA'
                ))
                
                # Pivot
                fig.add_hline(
                    y=pivot,
                    line=dict(color="blue", width=2),
                    annotation_text=f"🎯 Pivot: {pivot:,.0f}",
                    annotation_position="right"
                )
                
                # Stop
                fig.add_hline(
                    y=stop,
                    line=dict(color="red", width=2),
                    annotation_text=f"🛑 Stop: {stop:,.0f}",
                    annotation_position="right"
                )
                
                # Target 계산
                target_2r = pivot + 2 * r_value
                target_3r = pivot + 3 * r_value
                
                if show_2r:
                    fig.add_hline(
                        y=target_2r,
                        line=dict(color="green", width=1, dash="dot"),
                        annotation_text=f"2R: {target_2r:,.0f}",
                        annotation_position="right"
                    )
                
                if show_3r:
                    fig.add_hline(
                        y=target_3r,
                        line=dict(color="green", width=1, dash="dash"),
                        annotation_text=f"3R: {target_3r:,.0f}",
                        annotation_position="right"
                    )
                
                fig.update_layout(
                    title=f"{code} - 자동 계산 결과",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 수익률 계산
                profit_2r = ((target_2r - pivot) / pivot) * 100
                profit_3r = ((target_3r - pivot) / pivot) * 100
                loss = ((stop - pivot) / pivot) * 100
                rr_ratio = profit_2r / abs(loss)
                
                st.info(f"""
                **📊 예상 시나리오**
                
                **손실 시:**
                - Stop 도달: {loss:.1f}% 손실
                - 금액: -{max_loss:,.0f}원
                
                **수익 시:**
                - 2R 도달: +{profit_2r:.1f}% (+{qty * (target_2r - pivot):,.0f}원)
                - 3R 도달: +{profit_3r:.1f}% (+{qty * (target_3r - pivot):,.0f}원)
                
                **위험:보상 비율:** 1:{rr_ratio:.1f}
                """)
                
                # 상세 가이드
                with st.expander("📋 매매 실행 가이드"):
                    st.markdown(f"""
                    ### 진입 조건
                    
                    1. **가격**: 현재가({current_price:,.0f}원) → Pivot({pivot:,.0f}원) 돌파
                    2. **거래량**: 평균 대비 40~50% 증가 확인
                    3. **타이밍**: 
                       - 장중 돌파: 당일 종가 매수
                       - 장 마감 후 돌파: 익일 재진입 확인
                    
                    ### 손절 규칙
                    
                    - **Stop 가격**: {stop:,.0f}원
                    - **손절폭**: {stop_pct}%
                    - **규칙**: 이 가격 이탈 시 **즉시** 전량 청산
                    - **예외**: 없음
                    
                    ### 익절 전략
                    
                    **1차 익절 (2R: {target_2r:,.0f}원)**
                    - 수량의 30% 익절
                    - 수익 확정: +{profit_2r:.1f}%
                    
                    **2차 익절 (3R: {target_3r:,.0f}원)**
                    - 수량의 추가 30% 익절
                    - 수익 확정: +{profit_3r:.1f}%
                    
                    **나머지 40%:**
                    - 50일선 -3% 이탈 시 전량 청산
                    - 또는 추세 꺾임 시 판단
                    
                    ### 주의사항
                    
                    - 이 계산은 **참고용**입니다
                    - 최종 판단은 본인의 책임
                    - 뉴스/공시 확인 필수
                    - 감정 배제, 기계적 실행
                    """)

# -------------------------------------------------
# 하단
# -------------------------------------------------
st.divider()
st.caption("""
**자동 계산 + 시각화 = 판단은 사람의 몫**
""")
