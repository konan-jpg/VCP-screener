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
- 타점별 Entry · Stop · R
- 신뢰도 점수 (같은 종목 내 비교용)
""")

st.caption("※ 점수는 '확률'이 아니라 같은 종목 내 타점 비교 목적입니다")

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(code):
    """주식 데이터 로딩"""
    end = datetime.now()
    start = end - timedelta(days=250)
    try:
        df = fdr.DataReader(code, start, end)
        return df if df is not None and len(df) > 120 else None
    except Exception as e:
        return None

# -------------------------------------------------
# 지표 계산
# -------------------------------------------------
def prepare_indicators(df):
    """기술적 지표 추가"""
    df = df.copy()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["VolAvg60"] = df["Volume"].rolling(60).mean()
    return df

# -------------------------------------------------
# 거래량 Dry-up 점수
# -------------------------------------------------
def volume_dry_score(df):
    """
    거래량 고갈 정도 (0.6 ~ 1.0)
    최근 3일 최소 vs 60일 평균
    """
    recent_min = df["Volume"].tail(3).min()
    avg60 = df["VolAvg60"].iloc[-1]

    if pd.isna(avg60) or avg60 == 0:
        return 0.6

    ratio = recent_min / avg60

    if ratio < 0.4:
        return 1.0
    elif ratio < 0.6:
        return 0.8
    else:
        return 0.6

# -------------------------------------------------
# 거리 가중치 (방향성 포함)
# -------------------------------------------------
def distance_weight(entry, current):
    """
    현재가 vs 진입가 거리
    - 이미 돌파: 페널티
    - 적당한 거리: 보너스
    """
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
    """
    4가지 타점 계산
    - 정석 VCP: 베이스 최고점
    - Cheat: 상단 1/3 진입
    - Low Cheat: 현재 핸들
    - Pullback: 돌파 후 재진입
    """
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
    if len(cheat_zone) > 0:
        cheat_entry = cheat_zone["High"].tail(20).max()
    else:
        cheat_entry = base_high * 0.98
    cheat_stop = cheat_entry * 0.96

    # 3. Low Cheat
    low_cheat_entry = recent["High"].tail(10).max()
    
    # Stop = max(50일선, 최근저점)
    ma50 = recent["MA50"].iloc[-1]
    structural_low = recent["Low"].tail(10).min()
    
    # NaN 체크
    if pd.isna(ma50):
        ma50 = structural_low * 0.98
    
    low_cheat_stop = max(ma50, structural_low)

    # 4. Pullback
    pullback_entry = base_high
    pullback_stop = base_high * 0.97  # 3% 손절

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
    """
    타점 신뢰도 (0~100)
    - 기본 50
    - 거래량 Dry-up: +25
    - 거리 가중: +15
    - 변동성 수축: +10
    - 구조 페널티/보너스
    """
    current = df["Close"].iloc[-1]
    r = entry - stop

    if r <= 0:
        return 0

    score = 50.0

    # 거래량
    score += volume_dry_score(df) * 25

    # 거리
    score += distance_weight(entry, current) * 15

    # 변동성 수축
    short_range = (df["High"].tail(10) - df["Low"].tail(10)).mean()
    long_range = (df["High"].tail(60) - df["Low"].tail(60)).mean()
    
    if long_range > 0 and short_range / long_range < 0.6:
        score += 10

    # 타점별 보정
    risk_pct = (entry - stop) / entry

    if entry_type == "Low Cheat":
        # 손절이 너무 얇으면 페널티
        if risk_pct < 0.025:  # 2.5% 미만
            score -= 10
        # 현재가와 가까우면 보너스
        if abs(entry - current) / current < 0.03:
            score += 5

    elif entry_type == "Cheat":
        score *= 0.95

    elif entry_type == "Pullback":
        # 아직 돌파 안 했으면 페널티
        if current < entry:
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
        - 가장 보수적
        
        **Cheat Entry**
        - Entry: 상단 1/3 고점
        - Stop: -4%
        - 선행 진입
        
        **Low Cheat**
        - Entry: 최근 10일 고점
        - Stop: max(50일선, 최근저점)
        - 가장 공격적 (구조적 손절)
        
        **Pullback**
        - Entry: 베이스 최고가
        - Stop: -3%
        - 돌파 후 재진입
        """)

with col_output:
    if not code:
        st.info("👈 VCP 종목 코드를 입력하세요")
        
        st.markdown("""
        ### 사용 전제
        
        이 계산기는 **VCP가 이미 완성된 종목**을 가정합니다:
        - 스캐너에서 발견한 종목
        - Stage 2 상승 추세
        - 변동성 수축 확인
        - 거래량 Dry-up
        
        ### 신뢰도 점수
        
        점수는 **같은 종목 내** 타점 비교용입니다:
        - 종목 간 비교 ❌
        - 타점 간 비교 ⭕
        - 확률이 아님
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

            # 결과 표시
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
            ⭐ **추천 타점**: {best['타점']}
            - 신뢰도: {best['_score']}점
            - 진입가: {best['진입가']}
            - 손절가: {best['손절가']}
            """)
            
            # 현재가 vs 추천 타점
            current_price = df["Close"].iloc[-1]
            recommended_entry = best["_entry"]
            dist_pct = ((recommended_entry - current_price) / current_price) * 100
            
            if dist_pct < -3:
                st.warning(f"⚠️ 이미 돌파됨 (현재가: {current_price:,.0f})")
            elif dist_pct > 10:
                st.info(f"💡 진입가까지 {dist_pct:.1f}% 거리")
            else:
                st.success(f"✅ 진입 대기 구간 ({dist_pct:+.1f}%)")

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

            fig.update_layout(
                height=600,
                title=f"{code} - VCP 다중 타점",
                xaxis_rangeslider_visible=False,
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("※ 본 도구는 의사결정 보조용이며 매매 책임은 사용자에게 있습니다")
