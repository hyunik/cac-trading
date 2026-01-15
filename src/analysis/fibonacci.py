"""
피보나치 되돌림 모듈
- 손절 발생 시 복구 모드용
- 0.5~0.618 구간 복구 진입
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class FibonacciLevel:
    """피보나치 레벨"""
    ratio: float
    price: float
    label: str


@dataclass
class FibonacciZone:
    """피보나치 진입 구간"""
    entry_zone_start: float   # 0.5 레벨
    entry_zone_end: float     # 0.618 레벨
    stop_loss: float          # 0.0 레벨 (파동 시작점)
    target_1: float           # 0.382 레벨
    target_2: float           # 0.236 레벨
    is_bullish: bool          # 상승 복구인지 여부


class FibonacciRetracement:
    """피보나치 되돌림 분석기"""
    
    # 주요 피보나치 비율
    LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def __init__(self):
        pass
    
    def calculate_levels(
        self, 
        swing_high: float, 
        swing_low: float,
        is_uptrend: bool = True
    ) -> List[FibonacciLevel]:
        """
        피보나치 레벨 계산
        
        Args:
            swing_high: 파동 고점
            swing_low: 파동 저점
            is_uptrend: 상승 추세면 True (되돌림 하락), 하락 추세면 False (되돌림 상승)
        
        Returns:
            각 피보나치 레벨 리스트
        """
        levels = []
        range_size = swing_high - swing_low
        
        for ratio in self.LEVELS:
            if is_uptrend:
                # 상승 후 되돌림: 고점에서 아래로
                price = swing_high - (range_size * ratio)
            else:
                # 하락 후 되돌림: 저점에서 위로
                price = swing_low + (range_size * ratio)
            
            label = self._get_level_label(ratio)
            levels.append(FibonacciLevel(ratio=ratio, price=price, label=label))
        
        return levels
    
    def _get_level_label(self, ratio: float) -> str:
        """피보나치 비율에 대한 레이블"""
        labels = {
            0.0: "시작점 (0%)",
            0.236: "23.6%",
            0.382: "38.2%",
            0.5: "50.0% (복구 진입 시작)",
            0.618: "61.8% (황금 비율)",
            0.786: "78.6%",
            1.0: "끝점 (100%)"
        }
        return labels.get(ratio, f"{ratio*100:.1f}%")
    
    def find_swing_points(
        self, 
        df: pd.DataFrame, 
        lookback: int = 20
    ) -> Tuple[Optional[float], Optional[float], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        최근 파동의 고점/저점 찾기
        
        Args:
            df: OHLCV 데이터프레임
            lookback: 탐색 기간
        
        Returns:
            (swing_high, swing_low, high_time, low_time)
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent = df.tail(lookback)
        
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        swing_high = recent.loc[high_idx, 'high']
        swing_low = recent.loc[low_idx, 'low']
        
        return (swing_high, swing_low, high_idx, low_idx)
    
    def get_recovery_zone(
        self, 
        df: pd.DataFrame, 
        stop_loss_triggered_direction: str,  # "LONG" or "SHORT"
        lookback: int = 20
    ) -> Optional[FibonacciZone]:
        """
        손절 발생 후 복구 진입 구간 계산
        
        Args:
            df: OHLCV 데이터프레임
            stop_loss_triggered_direction: 손절된 포지션의 방향
            lookback: 파동 탐색 기간
        
        Returns:
            복구 진입 구간 (0.5~0.618)
        """
        swing_high, swing_low, high_time, low_time = self.find_swing_points(df, lookback)
        
        if swing_high is None or swing_low is None:
            return None
        
        range_size = swing_high - swing_low
        
        # 롱 손절 = 하락 추세 = 되돌림 상승에서 숏 진입
        # 숏 손절 = 상승 추세 = 되돌림 하락에서 롱 진입
        if stop_loss_triggered_direction == "LONG":
            # 하락 후 되돌림 상승: 저점에서 위로
            is_bullish = False  # 숏 복구
            entry_zone_start = swing_low + (range_size * 0.5)   # 0.5
            entry_zone_end = swing_low + (range_size * 0.618)   # 0.618
            stop_loss = swing_high  # 파동 고점 위
            target_1 = swing_low + (range_size * 0.382)
            target_2 = swing_low + (range_size * 0.236)
        else:
            # 상승 후 되돌림 하락: 고점에서 아래로
            is_bullish = True  # 롱 복구
            entry_zone_start = swing_high - (range_size * 0.5)   # 0.5
            entry_zone_end = swing_high - (range_size * 0.618)   # 0.618
            stop_loss = swing_low  # 파동 저점 아래
            target_1 = swing_high - (range_size * 0.382)
            target_2 = swing_high - (range_size * 0.236)
        
        return FibonacciZone(
            entry_zone_start=entry_zone_start,
            entry_zone_end=entry_zone_end,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            is_bullish=is_bullish
        )
    
    def is_in_recovery_zone(
        self, 
        current_price: float, 
        zone: FibonacciZone
    ) -> bool:
        """현재 가격이 복구 진입 구간에 있는지 확인"""
        if zone.is_bullish:
            # 롱 복구: 가격이 내려와야 함
            return zone.entry_zone_end <= current_price <= zone.entry_zone_start
        else:
            # 숏 복구: 가격이 올라와야 함
            return zone.entry_zone_start <= current_price <= zone.entry_zone_end
    
    def calculate_recovery_risk_reward(self, zone: FibonacciZone, entry_price: float) -> dict:
        """
        복구 진입의 리스크/리워드 계산
        
        Returns:
            {
                'risk': 손절까지 거리 (%),
                'reward_1': 1차 목표까지 이익 (%),
                'reward_2': 2차 목표까지 이익 (%),
                'risk_reward_ratio': 리스크 대비 리워드 비율
            }
        """
        if zone.is_bullish:
            risk = abs(entry_price - zone.stop_loss) / entry_price * 100
            reward_1 = abs(zone.target_1 - entry_price) / entry_price * 100
            reward_2 = abs(zone.target_2 - entry_price) / entry_price * 100
        else:
            risk = abs(zone.stop_loss - entry_price) / entry_price * 100
            reward_1 = abs(entry_price - zone.target_1) / entry_price * 100
            reward_2 = abs(entry_price - zone.target_2) / entry_price * 100
        
        return {
            'risk': risk,
            'reward_1': reward_1,
            'reward_2': reward_2,
            'risk_reward_ratio': reward_1 / risk if risk > 0 else 0
        }
