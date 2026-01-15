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
""")

# -------------------------------------------------
# 입력 UI
# -------------------------------------------------
col_input, col_chart = st.columns([1, 3])

with col_input:
    st.subheader("📥 입력")

    code = st.text_input("종목 코드 또는 이름", placeholder="예: 005930")

    pivot = st.number_input("Pivot", min_value=0.0, step=10.0)
    stop = st.number_input("Stop", min_value=0.0, step=10.0)

    show_2r = st.checkbox("2R 표시", value=True)
    show_3r = st.checkbox("3R 표시", value=True)

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
def load_data(code):
    end = datetime.now()
    start = end - timedelta(days=180)
    try:
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 0 else None
    except:
        return None

# -------------------------------------------------
# 차트 표시
# -------------------------------------------------
with col_chart:
    if not code:
        st.info("👈 종목을 입력하세요")
    else:
        df = load_data(code)

        if df is None:
            st.error("❌ 종목 데이터를 불러올 수 없습니다")
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

            # Pivot / Stop 검증
            if pivot > 0 and stop > 0 and pivot > stop:
                r = pivot - stop
                target_2r = pivot + 2 * r
                target_3r = pivot + 3 * r

                # Pivot
                fig.add_hline(
                    y=pivot,
                    line=dict(color="blue", width=2),
                    annotation_text="Pivot",
                    annotation_position="right"
                )

                # Stop
                fig.add_hline(
                    y=stop,
                    line=dict(color="red", width=2),
                    annotation_text="Stop",
                    annotation_position="right"
                )

                # Targets
                if show_2r:
                    fig.add_hline(
                        y=target_2r,
                        line=dict(color="green", width=1, dash="dot"),
                        annotation_text="2R",
                        annotation_position="right"
                    )

                if show_3r:
                    fig.add_hline(
                        y=target_3r,
                        line=dict(color="green", width=1, dash="dash"),
                        annotation_text="3R",
                        annotation_position="right"
                    )
            else:
                st.warning("Pivot과 Stop을 올바르게 입력하세요")

            fig.update_layout(
                title=f"{code} - Pivot / Stop / Target",
                height=700,
                xaxis_rangeslider_visible=False,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 하단 고정 문구
# -------------------------------------------------
st.divider()
st.caption("""
이 화면에서 하는 일은 하나뿐이다.  
**선을 보고, 판단은 사람의 몫으로 남긴다.**
""")
