"""
볼린저밴드 분석 모듈
- 20기간 이동평균 + 표준편차 2배
- 상/하단선 터치 감지
- 밴드 수축/확장 분석
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class BandPosition(Enum):
    """가격의 밴드 내 위치"""
    ABOVE_UPPER = "상단 돌파"
    TOUCH_UPPER = "상단 터치"
    MIDDLE_UPPER = "중앙 위"
    MIDDLE = "중앙"
    MIDDLE_LOWER = "중앙 아래"
    TOUCH_LOWER = "하단 터치"
    BELOW_LOWER = "하단 돌파"


@dataclass
class BollingerBandData:
    """볼린저밴드 데이터"""
    upper: float          # 상단선
    middle: float         # 중앙선 (20 MA)
    lower: float          # 하단선
    bandwidth: float      # 밴드폭 (%)
    percent_b: float      # %B (현재 가격의 상대 위치)
    position: BandPosition


class BollingerBands:
    """볼린저밴드 분석기"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, touch_threshold: float = 0.02):
        """
        Args:
            period: 이동평균 기간 (기본 20)
            std_dev: 표준편차 배수 (기본 2.0)
            touch_threshold: 터치 판정 임계값 (기본 2% - 밴드에서 2% 이내면 터치)
        """
        self.period = period
        self.std_dev = std_dev
        self.touch_threshold = touch_threshold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        볼린저밴드 계산
        
        Args:
            df: OHLCV 데이터프레임 (columns: close)
        
        Returns:
            볼린저밴드 컬럼이 추가된 데이터프레임
        """
        result = df.copy()
        
        # 중앙선: 단순 이동평균
        result['bb_middle'] = result['close'].rolling(window=self.period).mean()
        
        # 표준편차
        rolling_std = result['close'].rolling(window=self.period).std()
        
        # 상/하단선
        result['bb_upper'] = result['bb_middle'] + (rolling_std * self.std_dev)
        result['bb_lower'] = result['bb_middle'] - (rolling_std * self.std_dev)
        
        # 밴드폭 (%): (상단 - 하단) / 중앙 * 100
        result['bb_bandwidth'] = ((result['bb_upper'] - result['bb_lower']) / result['bb_middle']) * 100
        
        # %B: (현재가 - 하단) / (상단 - 하단)
        # 0 = 하단선, 1 = 상단선, 0.5 = 중앙
        band_range = result['bb_upper'] - result['bb_lower']
        result['bb_percent_b'] = np.where(
            band_range > 0,
            (result['close'] - result['bb_lower']) / band_range,
            0.5
        )
        
        return result
    
    def get_current_band(self, df: pd.DataFrame) -> Optional[BollingerBandData]:
        """현재(마지막) 캔들의 볼린저밴드 정보 반환"""
        calculated = self.calculate(df)
        
        if len(calculated) == 0 or pd.isna(calculated['bb_middle'].iloc[-1]):
            return None
        
        last = calculated.iloc[-1]
        
        return BollingerBandData(
            upper=last['bb_upper'],
            middle=last['bb_middle'],
            lower=last['bb_lower'],
            bandwidth=last['bb_bandwidth'],
            percent_b=last['bb_percent_b'],
            position=self._get_position(last['bb_percent_b'])
        )
    
    def _get_position(self, percent_b: float) -> BandPosition:
        """%B 값으로 밴드 내 위치 판정"""
        if percent_b > 1.0:
            return BandPosition.ABOVE_UPPER
        elif percent_b >= 1.0 - self.touch_threshold:
            return BandPosition.TOUCH_UPPER
        elif percent_b > 0.5:
            return BandPosition.MIDDLE_UPPER
        elif percent_b == 0.5:
            return BandPosition.MIDDLE
        elif percent_b > self.touch_threshold:
            return BandPosition.MIDDLE_LOWER
        elif percent_b >= 0.0:
            return BandPosition.TOUCH_LOWER
        else:
            return BandPosition.BELOW_LOWER
    
    def is_touching_upper(self, df: pd.DataFrame) -> bool:
        """상단선 터치 여부"""
        band = self.get_current_band(df)
        if band is None:
            return False
        return band.position in [BandPosition.TOUCH_UPPER, BandPosition.ABOVE_UPPER]
    
    def is_touching_lower(self, df: pd.DataFrame) -> bool:
        """하단선 터치 여부"""
        band = self.get_current_band(df)
        if band is None:
            return False
        return band.position in [BandPosition.TOUCH_LOWER, BandPosition.BELOW_LOWER]
    
    def is_band_squeeze(self, df: pd.DataFrame, squeeze_threshold: float = 4.0) -> bool:
        """
        밴드 수축(스퀴즈) 감지
        
        Args:
            df: OHLCV 데이터프레임
            squeeze_threshold: 밴드폭 임계값 (기본 4%)
        
        Returns:
            밴드폭이 임계값 이하이면 True
        """
        band = self.get_current_band(df)
        if band is None:
            return False
        return band.bandwidth <= squeeze_threshold
    
    def is_band_expansion(self, df: pd.DataFrame, lookback: int = 5) -> bool:
        """
        밴드 확장 감지 (최근 N개 캔들 대비)
        
        Args:
            df: OHLCV 데이터프레임
            lookback: 비교 기간
        
        Returns:
            현재 밴드폭이 최근 평균보다 크면 True
        """
        calculated = self.calculate(df)
        
        if len(calculated) < lookback + 1:
            return False
        
        recent_avg = calculated['bb_bandwidth'].iloc[-(lookback+1):-1].mean()
        current = calculated['bb_bandwidth'].iloc[-1]
        
        return current > recent_avg * 1.2  # 20% 이상 확장
    
    def get_band_touch_signal(self, df: pd.DataFrame) -> Tuple[bool, str, str]:
        """
        1시간봉 전략을 위한 볼린저밴드 터치 시그널
        
        Returns:
            (신호 발생 여부, 방향 "LONG"/"SHORT", 설명)
        """
        band = self.get_current_band(df)
        
        if band is None:
            return (False, "", "볼린저밴드 계산 불가")
        
        if band.position == BandPosition.TOUCH_LOWER or band.position == BandPosition.BELOW_LOWER:
            return (
                True,
                "LONG",
                f"볼린저밴드 하단 터치 - 역추세 롱 진입 조건 (%%B: {band.percent_b:.2f})"
            )
        
        elif band.position == BandPosition.TOUCH_UPPER or band.position == BandPosition.ABOVE_UPPER:
            return (
                True,
                "SHORT",
                f"볼린저밴드 상단 터치 - 역추세 숏 진입 조건 (%%B: {band.percent_b:.2f})"
            )
        
        return (False, "", f"볼린저밴드 중앙 구간 (%%B: {band.percent_b:.2f})")
