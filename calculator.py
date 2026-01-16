import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="VCP Multi-Entry Calculator", layout="wide")
st.title("🎯 VCP 다중 타점 계산기")

st.markdown("""
**VCP 완성 종목 전용 · 4가지 타점 자동 분석**

- 정석 VCP / Cheat / Low Cheat / Pullback
- 타점별 Entry · Stop · R 자동 계산
- 신뢰도 점수 (같은 종목 내 비교용)
- ✅ 20일 ATR (변동성 참고용)
""")

st.caption("※ ATR은 참고용 정보이며 매수·손절 계산에는 직접 사용되지 않습니다")

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(code):
    """주식 데이터 로딩"""
    end = datetime.now()
    start = end - timedelta(days=260)
    try:
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 120 else None
    except:
        return None

# -------------------------------------------------
# 지표 계산
# -------------------------------------------------
def prepare_indicators(df):
    """기술적 지표 + ATR 계산"""
    df = df.copy()

    # 이동평균
    df["MA50"] = df["Close"].rolling(50).mean()
    df["VolAvg60"] = df["Volume"].rolling(60).mean()

    # ATR(20) 계산
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],                # 당일 고저
        (df["High"] - prev_close).abs(),       # 전일종가-당일고가
        (df["Low"] - prev_close).abs()         # 전일종가-당일저가
    ], axis=1).max(axis=1)

    df["ATR20"] = tr.rolling(20).mean()

    return df

# -------------------------------------------------
# 거래량 Dry-up 점수
# -------------------------------------------------
def volume_dry_score(df):
    """거래량 고갈 정도 (0.6 ~ 1.0)"""
    recent_min = df["Volume"].tail(3).min()
    avg60 = df["VolAvg60"].iloc[-1]

    if pd.isna(avg60) or avg60 == 0:
        return 0.6

    ratio = recent_min / avg60
    if ratio < 0.4:
        return 1.0
    elif ratio < 0.6:
        return 0.8
    return 0.6

# -------------------------------------------------
# 거리 가중치
# -------------------------------------------------
def distance_weight(entry, current):
    """현재가 vs 진입가 거리 가중"""
    if entry == 0:
        return 0.5

    dist_pct = ((entry - current) / current) * 100

    if dist_pct < -3:
        return 0.5        # 이미 돌파
    if 0 <= dist_pct < 2:
        return 1.0        # 거의 도달
    if 2 <= dist_pct < 5:
        return 0.95       # 이상적
    if 5 <= dist_pct < 8:
        return 0.85       # 약간 멀음
    return 0.7            # 너무 멀음

# -------------------------------------------------
# 타점 계산
# -------------------------------------------------
def calculate_entries(df):
    """4가지 타점 계산"""
    recent = df.tail(120)

    base_high = recent["High"].max()
    base_low = recent["Low"].min()
    base_range = base_high - base_low
    upper_third = base_low + base_range * 0.66

    # 1. 정석 VCP
    vcp_entry = base_high
    vcp_stop = base_high * 0.95

    # 2. Cheat Entry
    cheat_zone = recent[recent["High"] >= upper_third]
    cheat_entry = cheat_zone["High"].tail(20).max() if len(cheat_zone) else base_high * 0.98
    cheat_stop = cheat_entry * 0.96

    # 3. Low Cheat
    low_cheat_entry = recent["High"].tail(10).max()
    ma50 = recent["MA50"].iloc[-1]
    structural_low = recent["Low"].tail(10).min()

    if pd.isna(ma50):
        ma50 = structural_low * 0.98

    low_cheat_stop = max(ma50, structural_low)

    # 4. Pullback
    pullback_entry = base_high
    pullback_stop = base_high * 0.97

    return {
        "정석 VCP": (vcp_entry, vcp_stop),
        "Cheat": (cheat_entry, cheat_stop),
        "Low Cheat": (low_cheat_entry, low_cheat_stop),
        "Pullback": (pullback_entry, pullback_stop)
    }

# -------------------------------------------------
# 신뢰 점수
# -------------------------------------------------
def confidence_score(entry, stop, df, entry_type):
    """타점 신뢰도 (0~100)"""
    current = df["Close"].iloc[-1]
    r = entry - stop

    if r <= 0:
        return 0

    score = 50.0
    score += volume_dry_score(df) * 25
    score += distance_weight(entry, current) * 15

    short_range = (df["High"].tail(10) - df["Low"].tail(10)).mean()
    long_range = (df["High"].tail(60) - df["Low"].tail(60)).mean()
    if long_range > 0 and short_range / long_range < 0.6:
        score += 10

    risk_pct = (entry - stop) / entry

    if entry_type == "Low Cheat":
        if risk_pct < 0.025:
            score -= 10
        if abs(entry - current) / current < 0.03:
            score += 5

    elif entry_type == "Cheat":
        score *= 0.95

    elif entry_type == "Pullback" and current < entry:
        score *= 0.8

    return min(int(score), 100)

# -------------------------------------------------
# UI
# -------------------------------------------------
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📥 입력")
    
    code = st.text_input(
        "종목 코드",
        placeholder="예: 005930",
        help="VCP 패턴이 완성된 종목"
    )
    
    st.divider()
    
    with st.expander("💡 타점 설명"):
        st.markdown("""
        **정석 VCP**
        - Entry: 베이스 최고가
        - Stop: -5%
        
        **Cheat Entry**
        - Entry: 상단 1/3 고점
        - Stop: -4%
        
        **Low Cheat**
        - Entry: 최근 10일 고점
        - Stop: max(50일선, 최근저점)
        
        **Pullback**
        - Entry: 베이스 최고가
        - Stop: -3%
        """)

with col_output:
    if not code:
        st.info("👈 VCP 종목 코드를 입력하세요")
        
        st.markdown("""
        ### 📐 ATR(Average True Range)이란?
        
        **정의:**
        - 일정 기간 동안의 평균 변동폭
        - 20일 ATR = 최근 20일 평균 변동성
        
        **VCP 관점:**
        - ATR 낮음 = 조용함 = VCP 이상적
        - ATR 높음 = 변동성 큼 = 위험
        
        **사용법:**
        - 이 계산기는 ATR을 **참고만** 합니다
        - 손절/진입 계산은 **구조 우선**
        - ATR은 변동성 컨텍스트 제공
        """)
    else:
        df = load_data(code)

        if df is None:
            st.error("❌ 데이터 로딩 실패")
            st.info("종목 코드를 확인하세요")
        else:
            # 지표 계산
            df = prepare_indicators(df)
            
            # 타점 계산
            entries = calculate_entries(df)

            # 결과 테이블
            rows = []
            for name, (entry, stop) in entries.items():
                score = confidence_score(entry, stop, df, name)
                r_value = entry - stop
                
                rows.append({
                    "타점": name,
                    "진입가": f"{entry:,.0f}",
                    "손절가": f"{stop:,.0f}",
                    "R": f"{r_value:,.0f}",
                    "손절폭": f"{(stop-entry)/entry*100:.1f}%",
                    "신뢰도": score,
                    "_entry": entry,
                    "_stop": stop,
                    "_score": score
                })

            df_result = pd.DataFrame(rows).sort_values("_score", ascending=False)
            df_result.insert(0, "순위", range(1, len(df_result) + 1))

            # 타점 테이블 표시
            st.subheader("📊 타점 비교 (신뢰도 순)")
            
            display_cols = ["순위","타점","진입가","손절가","R","손절폭","신뢰도"]
            st.dataframe(
                df_result[display_cols],
                use_container_width=True,
                hide_index=True
            )

            # 추천 타점
            best = df_result.iloc[0]
            st.success(f"""
            ⭐ **자동 추천 타점**: {best['타점']}
            - 신뢰도: {best['_score']}점
            - 진입가: {best['진입가']}
            - 손절가: {best['손절가']}
            - R: {best['R']}
            """)

            # 현재가 정보
            current_price = df["Close"].iloc[-1]
            recommended_entry = best["_entry"]
            dist_pct = ((recommended_entry - current_price) / current_price) * 100

            if dist_pct < -3:
                st.warning(f"⚠️ 이미 돌파됨 (현재가: {current_price:,.0f})")
            elif dist_pct > 10:
                st.info(f"💡 진입가까지 {dist_pct:.1f}% 떨어져 있음")
            else:
                st.success(f"✅ 진입 대기 구간 (현재가: {current_price:,.0f}, {dist_pct:+.1f}%)")

            # ATR 정보
            st.divider()
            st.markdown("### 📐 변동성 (ATR 20일)")
            
            atr20 = df["ATR20"].iloc[-1]

            if not pd.isna(atr20):
                atr_pct = atr20 / current_price * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("ATR(20)", f"{atr20:,.0f}원")
                col2.metric("ATR / 현재가", f"{atr_pct:.2f}%")

                with col3:
                    if atr_pct < 2:
                        st.success("✅ 매우 조용함 (VCP 이상적)")
                    elif atr_pct < 4:
                        st.info("ℹ️ 정상 범위")
                    else:
                        st.warning("⚠️ 변동성 높음 (주의)")

                st.caption(f"💡 ATR은 최근 20일 평균 변동폭입니다. 낮을수록 VCP에 적합합니다.")
            else:
                st.warning("ATR 계산 불가")

            # 차트
            st.divider()
            st.markdown("### 📈 차트")
            
            fig = go.Figure()
            chart_df = df.tail(120)

            # 캔들
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Price"
            ))

            # 50일선
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df["MA50"],
                name="50MA",
                line=dict(color="blue", dash="dot")
            ))

            # 타점 라인
            for _, r in df_result.iterrows():
                color = "gold" if r["순위"] == 1 else "gray"
                width = 2 if r["순위"] == 1 else 1
                
                fig.add_hline(
                    y=r["_entry"],
                    line=dict(color=color, dash="dot", width=width),
                    annotation_text=f"{r['타점']} 진입"
                )
                fig.add_hline(
                    y=r["_stop"],
                    line=dict(color="red", dash="dash", width=1),
                    annotation_text=f"{r['타점']} 손절"
                )

            # 제목에 ATR 포함
            title_text = f"{code} - VCP 다중 타점"
            if not pd.isna(atr20):
                title_text += f" | ATR20: {atr20:,.0f}원 ({atr_pct:.1f}%)"

            fig.update_layout(
                height=600,
                title=title_text,
                xaxis_rangeslider_visible=False,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("※ ATR은 변동성 참고용이며 매매 기준은 구조가 우선입니다")
