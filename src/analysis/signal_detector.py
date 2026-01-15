"""
시그널 캔들 감지 모듈
- 장악형 캔들 (Engulfing)
- 스타형 캔들 (Morning/Evening Star)
- 망치형/역망치형 (Hammer/Inverted Hammer)
- 도지형 (Doji)
- 거래량 검증
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


class SignalType(Enum):
    """시그널 유형"""
    BULLISH_ENGULFING = "상승장악형"
    BEARISH_ENGULFING = "하락장악형"
    MORNING_STAR = "샛별형"
    EVENING_STAR = "석별형"
    HAMMER = "망치형"
    INVERTED_HAMMER = "역망치형"
    HANGING_MAN = "교수형"
    SHOOTING_STAR = "유성형"
    BULLISH_DOJI = "상승도지"
    BEARISH_DOJI = "하락도지"


class SignalConfidence(Enum):
    """시그널 신뢰도"""
    HIGH = 5      # ★★★★★ 장악형, 스타형
    MEDIUM = 3    # ★★★ 망치형, 도지형
    LOW = 1       # ★ 단독 시그널


@dataclass
class CandleSignal:
    """캔들 시그널 정보"""
    signal_type: SignalType
    confidence: SignalConfidence
    timestamp: pd.Timestamp
    entry_price: float           # 권장 진입가
    stop_loss: float             # 손절가 (시그널 캔들 꼬리 끝)
    volume_ratio: float          # 평균 거래량 대비 비율
    is_valid: bool               # 거래량 검증 통과 여부
    description: str             # 시그널 설명


class SignalDetector:
    """시그널 캔들 패턴 감지기"""
    
    def __init__(self, volume_threshold: float = 1.0):
        """
        Args:
            volume_threshold: 평균 거래량 대비 최소 비율 (기본 1.0 = 평균 이상)
        """
        self.volume_threshold = volume_threshold
        self.volume_ma_period = 20  # 거래량 이동평균 기간
    
    def detect_all_signals(self, df: pd.DataFrame) -> List[CandleSignal]:
        """
        모든 시그널 패턴 감지
        
        Args:
            df: OHLCV 데이터프레임 (columns: open, high, low, close, volume)
        
        Returns:
            감지된 시그널 리스트
        """
        signals = []
        
        if len(df) < 3:
            return signals
        
        # 거래량 이동평균 계산
        df = df.copy()
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 캔들 특성 계산
        df['body'] = df['close'] - df['open']
        df['body_abs'] = abs(df['body'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        
        # 각 패턴 감지
        engulfing = self._detect_engulfing(df)
        stars = self._detect_star_patterns(df)
        hammers = self._detect_hammer_patterns(df)
        dojis = self._detect_doji_patterns(df)
        
        signals.extend(engulfing)
        signals.extend(stars)
        signals.extend(hammers)
        signals.extend(dojis)
        
        return signals
    
    def _detect_engulfing(self, df: pd.DataFrame) -> List[CandleSignal]:
        """
        장악형 캔들 감지 (신뢰도 ★★★★★)
        - 현재 캔들 몸통이 직전 캔들 몸통을 완전히 덮음
        - 직전 캔들과 색깔이 다름
        """
        signals = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            # 직전 캔들 몸통이 너무 작으면 스킵 (도지형 제외)
            if prev['body_abs'] < (prev['high'] - prev['low']) * 0.1:
                continue
            
            prev_body_top = max(prev['open'], prev['close'])
            prev_body_bottom = min(prev['open'], prev['close'])
            curr_body_top = max(curr['open'], curr['close'])
            curr_body_bottom = min(curr['open'], curr['close'])
            
            volume_ratio = curr['volume_ratio'] if pd.notna(curr['volume_ratio']) else 1.0
            is_volume_valid = volume_ratio >= self.volume_threshold
            
            # 상승 장악형: 음봉 -> 양봉, 현재 몸통이 직전 몸통 포함
            if (not prev['is_bullish'] and curr['is_bullish'] and 
                curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom):
                
                # 진입가: 몸통의 40~50% 부근 조정 시
                entry_price = curr_body_bottom + (curr['body_abs'] * 0.45)
                stop_loss = curr['low']
                
                signals.append(CandleSignal(
                    signal_type=SignalType.BULLISH_ENGULFING,
                    confidence=SignalConfidence.HIGH,
                    timestamp=curr.name if isinstance(curr.name, pd.Timestamp) else pd.Timestamp(curr.name),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"상승장악형: 직전 음봉을 완전히 덮는 강한 양봉 (거래량 {volume_ratio:.1f}배)"
                ))
            
            # 하락 장악형: 양봉 -> 음봉, 현재 몸통이 직전 몸통 포함
            elif (prev['is_bullish'] and not curr['is_bullish'] and 
                  curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom):
                
                entry_price = curr_body_top - (curr['body_abs'] * 0.45)
                stop_loss = curr['high']
                
                signals.append(CandleSignal(
                    signal_type=SignalType.BEARISH_ENGULFING,
                    confidence=SignalConfidence.HIGH,
                    timestamp=curr.name if isinstance(curr.name, pd.Timestamp) else pd.Timestamp(curr.name),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"하락장악형: 직전 양봉을 완전히 덮는 강한 음봉 (거래량 {volume_ratio:.1f}배)"
                ))
        
        return signals
    
    def _detect_star_patterns(self, df: pd.DataFrame) -> List[CandleSignal]:
        """
        스타형 캔들 감지 (신뢰도 ★★★★★)
        - 3개 캔들 조합: 추세봉 - 도지/망치 - 반전봉
        - Morning Star: 하락 -> 도지 -> 상승
        - Evening Star: 상승 -> 도지 -> 하락
        """
        signals = []
        
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # 2번 캔들은 몸통이 작아야 함 (도지 또는 망치)
            second_range = second['high'] - second['low']
            if second_range == 0 or second['body_abs'] / second_range > 0.3:
                continue
            
            volume_ratio = third['volume_ratio'] if pd.notna(third['volume_ratio']) else 1.0
            is_volume_valid = volume_ratio >= self.volume_threshold
            
            first_body = first['body_abs']
            third_body = third['body_abs']
            
            # Morning Star: 강한 하락 -> 도지 -> 강한 상승
            if (not first['is_bullish'] and third['is_bullish'] and
                first_body > third_body * 0.5 and  # 1번 캔들이 어느정도 크기
                third['close'] > (first['open'] + first['close']) / 2):  # 3번이 1번 중간 이상
                
                entry_price = third['close']
                stop_loss = min(first['low'], second['low'], third['low'])
                
                signals.append(CandleSignal(
                    signal_type=SignalType.MORNING_STAR,
                    confidence=SignalConfidence.HIGH,
                    timestamp=third.name if isinstance(third.name, pd.Timestamp) else pd.Timestamp(third.name),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"샛별형: 하락 추세 후 반전 신호 (거래량 {volume_ratio:.1f}배)"
                ))
            
            # Evening Star: 강한 상승 -> 도지 -> 강한 하락
            elif (first['is_bullish'] and not third['is_bullish'] and
                  first_body > third_body * 0.5 and
                  third['close'] < (first['open'] + first['close']) / 2):
                
                entry_price = third['close']
                stop_loss = max(first['high'], second['high'], third['high'])
                
                signals.append(CandleSignal(
                    signal_type=SignalType.EVENING_STAR,
                    confidence=SignalConfidence.HIGH,
                    timestamp=third.name if isinstance(third.name, pd.Timestamp) else pd.Timestamp(third.name),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"석별형: 상승 추세 후 반전 신호 (거래량 {volume_ratio:.1f}배)"
                ))
        
        return signals
    
    def _detect_hammer_patterns(self, df: pd.DataFrame) -> List[CandleSignal]:
        """
        망치형/역망치형 캔들 감지 (신뢰도 ★★★)
        - 꼬리 길이가 몸통의 2배 이상
        - 진입: 꼬리 길이의 35% 지점
        """
        signals = []
        
        for i in range(len(df)):
            candle = df.iloc[i]
            
            body = candle['body_abs']
            if body == 0:
                body = 0.0001  # 도지 처리
            
            upper_wick = candle['upper_wick']
            lower_wick = candle['lower_wick']
            
            volume_ratio = candle['volume_ratio'] if pd.notna(candle['volume_ratio']) else 1.0
            is_volume_valid = volume_ratio >= self.volume_threshold
            
            timestamp = candle.name if isinstance(candle.name, pd.Timestamp) else pd.Timestamp(candle.name)
            
            # 망치형 (Hammer): 긴 아래꼬리, 짧은 윗꼬리 - 하락 중 상승 반전
            if lower_wick >= body * 2 and upper_wick <= body * 0.5:
                # 진입가: 꼬리 길이의 35% 지점
                entry_price = candle['low'] + (lower_wick * 0.35)
                stop_loss = candle['low']
                
                signals.append(CandleSignal(
                    signal_type=SignalType.HAMMER,
                    confidence=SignalConfidence.MEDIUM,
                    timestamp=timestamp,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"망치형: 긴 아래꼬리({lower_wick/body:.1f}배) 상승 반전 신호 (거래량 {volume_ratio:.1f}배)"
                ))
            
            # 역망치형 (Inverted Hammer): 긴 윗꼬리, 짧은 아래꼬리 - 하락 중 상승 반전
            elif upper_wick >= body * 2 and lower_wick <= body * 0.5:
                entry_price = candle['high'] - (upper_wick * 0.35)
                stop_loss = candle['low']
                
                signals.append(CandleSignal(
                    signal_type=SignalType.INVERTED_HAMMER,
                    confidence=SignalConfidence.MEDIUM,
                    timestamp=timestamp,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    volume_ratio=volume_ratio,
                    is_valid=is_volume_valid,
                    description=f"역망치형: 긴 윗꼬리({upper_wick/body:.1f}배) 상승 반전 신호 (거래량 {volume_ratio:.1f}배)"
                ))
            
            # 교수형 (Hanging Man): 상승 중 긴 아래꼬리 - 하락 반전
            # 유성형 (Shooting Star): 상승 중 긴 윗꼬리 - 하락 반전
            # (추세 판단은 전략 모듈에서 처리)
        
        return signals
    
    def _detect_doji_patterns(self, df: pd.DataFrame) -> List[CandleSignal]:
        """
        도지형 캔들 감지 (신뢰도 ★★★)
        - 시가와 종가가 거의 일치 (몸통이 전체 범위의 10% 이하)
        - 볼린저밴드 상/하단에서 연속 2개 발생 시 강력 신호
        """
        signals = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            curr_range = curr['high'] - curr['low']
            prev_range = prev['high'] - prev['low']
            
            if curr_range == 0:
                continue
            
            # 도지 판정: 몸통이 전체 범위의 10% 이하
            is_curr_doji = curr['body_abs'] / curr_range <= 0.1
            is_prev_doji = prev_range > 0 and prev['body_abs'] / prev_range <= 0.1
            
            if not is_curr_doji:
                continue
            
            volume_ratio = curr['volume_ratio'] if pd.notna(curr['volume_ratio']) else 1.0
            is_volume_valid = volume_ratio >= self.volume_threshold
            
            timestamp = curr.name if isinstance(curr.name, pd.Timestamp) else pd.Timestamp(curr.name)
            
            # 연속 도지는 더 강력한 신호
            confidence = SignalConfidence.HIGH if is_prev_doji else SignalConfidence.MEDIUM
            description_suffix = " (연속 도지 - 강력 신호)" if is_prev_doji else ""
            
            # 위치에 따른 방향 결정 (볼린저밴드는 전략 모듈에서 판단)
            # 여기서는 일단 도지 발생만 기록
            mid_price = (curr['high'] + curr['low']) / 2
            
            signals.append(CandleSignal(
                signal_type=SignalType.BULLISH_DOJI,  # 방향은 전략에서 결정
                confidence=confidence,
                timestamp=timestamp,
                entry_price=mid_price,
                stop_loss=curr['low'],
                volume_ratio=volume_ratio,
                is_valid=is_volume_valid,
                description=f"도지형: 추세 전환 가능성{description_suffix} (거래량 {volume_ratio:.1f}배)"
            ))
        
        return signals
    
    def filter_valid_signals(self, signals: List[CandleSignal]) -> List[CandleSignal]:
        """거래량 검증을 통과한 유효 시그널만 필터링"""
        return [s for s in signals if s.is_valid]
    
    def get_high_confidence_signals(self, signals: List[CandleSignal]) -> List[CandleSignal]:
        """고신뢰도 시그널만 필터링 (★★★★★)"""
        return [s for s in signals if s.confidence == SignalConfidence.HIGH]
