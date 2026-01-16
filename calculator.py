def find_tight_zone_low(df, lookback=20, max_days=10):
    """
    마지막 타이트 구간(변동성 수축)의 저점 찾기
    - 최근 lookback일 중에서
    - 연속된 max_days일 이내에
    - 일일 변동폭(High-Low)이 ATR 대비 작은 구간
    의 최저점 반환
    """
    recent = df.tail(lookback)
    atr = recent['ATR20'].iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        # fallback: 최근 5일
        return float(recent['Low'].tail(5).min())
    
    # 일일 변동폭 계산
    daily_range = recent['High'] - recent['Low']
    
    # 타이트 조건: 변동폭 < ATR * 0.6
    tight_mask = daily_range < (atr * 0.6)
    
    # 연속된 타이트 구간 찾기 (최근부터 역순)
    tight_indices = recent[tight_mask].index
    
    if len(tight_indices) == 0:
        # 타이트 구간 없으면 최근 5일 저점
        return float(recent['Low'].tail(5).min())
    
    # 마지막(가장 최근) 타이트 구간의 저점
    # 최근 max_days일 이내의 타이트 구간만
    last_tight = recent.loc[tight_indices].tail(max_days)
    
    return float(last_tight['Low'].min())

def calculate_entries(df, atr_buffer_mult=0.3):
    """4가지 진입타점 계산 (타이트 구간 기반 손절)"""
    recent = df.tail(120)
    atr20 = recent["ATR20"].iloc[-1]
    
    if pd.isna(atr20) or atr20 <= 0:
        atr20 = recent["Close"].iloc[-1] * 0.02
    
    buffer = atr_buffer_mult * atr20
    
    base_high = float(recent["High"].max())
    base_low = float(recent["Low"].min())
    base_range = base_high - base_low
    upper_third = base_low + base_range * 0.66
    
    current_price = float(recent["Close"].iloc[-1])
    
    # ========================================
    # 1) 정석 VCP
    # ========================================
    vcp_entry = base_high
    
    # 마지막 타이트 구간(최근 20일 중 변동성 낮은 구간)의 저점
    vcp_structure_low = find_tight_zone_low(df, lookback=20, max_days=10)
    vcp_stop = max(100.0, vcp_structure_low - buffer)
    
    # 리스크 체크: 10% 이상이면 경고용 조정
    vcp_risk = (vcp_entry - vcp_stop) / vcp_entry
    if vcp_risk > 0.10:
        # 너무 넓으면 강제로 -8%로 제한 (경고 표시 필요)
        vcp_stop = vcp_entry * 0.92
    
    # ========================================
    # 2) Cheat Entry
    # ========================================
    cheat_zone = recent[recent["High"] >= upper_third]
    
    if len(cheat_zone) > 0:
        cheat_entry = float(cheat_zone["High"].tail(20).max())
        # Cheat 구간의 타이트 존 저점
        cheat_tight_low = float(cheat_zone['Low'].tail(10).min())
        cheat_stop = max(100.0, cheat_tight_low - buffer)
        
        cheat_risk = (cheat_entry - cheat_stop) / cheat_entry
        if cheat_risk > 0.10:
            cheat_stop = cheat_entry * 0.92
    else:
        cheat_entry = base_high * 0.98
        cheat_stop = vcp_stop
    
    # ========================================
    # 3) Low Cheat (기존 로직 유지)
    # ========================================
    trigger = find_low_cheat_trigger(df, lookback=60)
    
    if trigger is not None and not pd.isna(trigger["ATR20"]):
        low_entry = float(trigger["High"])
        low_stop = max(100.0, float(trigger["Low"] - atr_buffer_mult * trigger["ATR20"]))
    else:
        # fallback: 최근 타이트 구간
        low_entry = float(recent["High"].tail(10).max())
        low_tight_low = find_tight_zone_low(df, lookback=15, max_days=7)
        low_stop = max(100.0, low_tight_low - buffer)
    
    low_risk = (low_entry - low_stop) / low_entry
    if low_risk > 0.10:
        low_stop = low_entry * 0.92
    
    # ========================================
    # 4) Pullback
    # ========================================
    pull_entry = base_high
    
    # 풀백 = 돌파 후 타이트한 재테스트 구간
    pull_tight_low = find_tight_zone_low(df, lookback=15, max_days=7)
    pull_stop = max(100.0, pull_tight_low - buffer)
    
    pull_risk = (pull_entry - pull_stop) / pull_entry
    if pull_risk > 0.10:
        pull_stop = pull_entry * 0.92
    
    return {
        "정석 VCP": (vcp_entry, vcp_stop),
        "Cheat": (cheat_entry, cheat_stop),
        "Low Cheat": (low_entry, low_stop),
        "Pullback": (pull_entry, pull_stop),
    }


