import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="VCP Chart Target Viewer", layout="wide")
st.title("📈 VCP Target / Stop Chart Viewer")

st.markdown("""
**종목을 입력하면 Pivot · Stop · Target이 차트에 표시됩니다**

- 계산 ❌
- 판별 ❌
- 시각화 ⭕
- **KRX 접속 실패 시 백업 사용**
""")

# -------------------------------------------------
# 데이터 로딩 (백업 포함)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(code):
    """
    주식 데이터 로딩 (백업 포함)
    1. FinanceDataReader 시도
    2. 실패 시 경고 메시지
    """
    end = datetime.now()
    start = end - timedelta(days=180)
    
    try:
        df = fdr.DataReader(code, start, end)
        if df is not None and len(df) > 0:
            st.success(f"✅ {code} 데이터 로딩 성공")
            return df
        else:
            st.warning(f"⚠️ {code} 데이터가 비어있습니다")
            return None
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {str(e)}")
        st.info("💡 종목 코드를 확인하거나, KRX 서버 상태를 확인하세요")
        return None

# -------------------------------------------------
# 입력 UI
# -------------------------------------------------
col_input, col_chart = st.columns([1, 3])

with col_input:
    st.subheader("📥 입력")

    code = st.text_input(
        "종목 코드 또는 이름", 
        placeholder="예: 005930 또는 삼성전자",
        help="6자리 종목 코드를 입력하세요"
    )

    pivot = st.number_input(
        "Pivot (진입가)", 
        min_value=0.0, 
        step=100.0,
        help="차트에서 확인한 Pivot 가격"
    )
    
    stop = st.number_input(
        "Stop (손절가)", 
        min_value=0.0, 
        step=100.0,
        help="계획한 손절 가격"
    )

    st.divider()
    
    st.markdown("### 🎯 목표가 표시")
    show_2r = st.checkbox("2R (1차 목표)", value=True)
    show_3r = st.checkbox("3R (2차 목표)", value=True)

# -------------------------------------------------
# 차트 표시
# -------------------------------------------------
with col_chart:
    if not code:
        st.info("👈 종목 코드를 입력하세요")
        
        with st.expander("💡 사용 방법"):
            st.markdown("""
            ### 사용법
            
            1. **종목 코드 입력**
               - 6자리 숫자 (예: 005930)
               - 또는 종목명 (예: 삼성전자)
            
            2. **Pivot 입력**
               - 스캐너에서 찾은 종목의 최근 고점
               - 차트에서 육안으로 확인한 저항선
            
            3. **Stop 입력**
               - Pivot 기준 5~7% 하락한 가격
               - 예: Pivot 100,000원 → Stop 95,000원
            
            4. **R 배수 계산**
               - 1R = Pivot - Stop
               - 2R = Pivot + (1R × 2)
               - 3R = Pivot + (1R × 3)
            
            ### 백업 CSV
            
            KRX 서버 접속이 안 될 경우:
            - 백업 CSV 준비 필요 없음
            - 종목 코드만 정확하면 작동
            - 데이터는 FinanceDataReader가 처리
            """)
    else:
        df = load_data(code)

        if df is None:
            st.error("❌ 종목 데이터를 불러올 수 없습니다")
            st.markdown("""
            **문제 해결:**
            1. 종목 코드 확인 (6자리 숫자)
            2. KRX 서버 상태 확인
            3. 잠시 후 다시 시도
            """)
        else:
            fig = go.Figure()

            # 캔들
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ))
            
            # 50일 이평선 (참고)
            ma50 = df['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ma50,
                line=dict(color='blue', width=1, dash='dot'),
                name='50MA'
            ))

            # Pivot / Stop 검증 및 표시
            if pivot > 0 and stop > 0 and pivot > stop:
                r = pivot - stop
                target_2r = pivot + 2 * r
                target_3r = pivot + 3 * r

                # Pivot
                fig.add_hline(
                    y=pivot,
                    line=dict(color="blue", width=2),
                    annotation_text=f"Pivot: {pivot:,.0f}",
                    annotation_position="right"
                )

                # Stop
                fig.add_hline(
                    y=stop,
                    line=dict(color="red", width=2),
                    annotation_text=f"Stop: {stop:,.0f}",
                    annotation_position="right"
                )

                # 1R 계산 표시
                current_price = df['Close'].iloc[-1]
                
                st.info(f"""
                **R 계산:**
                - 1R = {r:,.0f}원
                - 현재가: {current_price:,.0f}원
                - Pivot까지: {((pivot - current_price) / current_price * 100):+.1f}%
                """)

                # Targets
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
                    
                # 수익률 계산
                profit_2r = ((target_2r - pivot) / pivot) * 100
                profit_3r = ((target_3r - pivot) / pivot) * 100
                loss = ((stop - pivot) / pivot) * 100
                
                st.success(f"""
                **예상 수익/손실:**
                - Stop 도달 시: {loss:.1f}%
                - 2R 도달 시: +{profit_2r:.1f}%
                - 3R 도달 시: +{profit_3r:.1f}%
                - 위험:보상 비율 = 1:{profit_2r/abs(loss):.1f} (2R 기준)
                """)
            else:
                st.warning("⚠️ Pivot과 Stop을 올바르게 입력하세요 (Pivot > Stop)")

            fig.update_layout(
                title=f"{code} - Pivot / Stop / Target 차트",
                height=700,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # 추가 정보
            with st.expander("📊 차트 해석 가이드"):
                st.markdown("""
                ### 진입 시점
                - 현재가가 Pivot을 **강한 거래량**과 함께 돌파
                - 돌파 거래량: 평균 대비 40~50% 이상
                - 당일 또는 익일 재진입 시점에서 매수
                
                ### 손절 규칙
                - Stop 가격 이탈 시 **즉시 청산**
                - 예외 없음
                - 감정 배제
                
                ### 익절 전략
                - 2R 도달: 30% 익절
                - 3R 도달: 추가 30% 익절
                - 나머지: 50일선 -3% 이탈 시 전량 청산
                
                ### 주의사항
                - 이 도구는 시각화만 제공
                - 최종 판단은 본인의 책임
                - 뉴스/공시 확인 필수
                """)

# -------------------------------------------------
# 하단
# -------------------------------------------------
st.divider()
st.caption("""
**이 화면에서 하는 일은 하나뿐이다.**  
선을 보고, 판단은 사람의 몫으로 남긴다.
""")
