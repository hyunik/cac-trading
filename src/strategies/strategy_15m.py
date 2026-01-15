"""
전략 A: 15분봉 리스크관리 매매법 (추세 피라미딩)

- 포지션 결정: 오전 9시 일봉 캔들 분석으로 데일리 바이어스 설정
- 진입 비중: 초기 시드의 10%
- 피라미딩: 최소 30% 수익 격차 발생 시 10% -> 20% -> 60% 순으로 추가 진입
- 탈출: 1차 수익 30% 달성 시 35% 물량 익절 후 본절 스탑로스 이동
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd

from ..analysis.signal_detector import SignalDetector, CandleSignal, SignalType


class DailyBias(Enum):
    """일봉 기준 데일리 바이어스"""
    BULLISH = "상승"
    BEARISH = "하락"
    NEUTRAL = "중립"


class PyramidLevel(Enum):
    """피라미딩 단계"""
    INITIAL = 1      # 초기 진입 10%
    LEVEL_2 = 2      # 2차 진입 20%
    LEVEL_3 = 3      # 3차 진입 60%


@dataclass
class Position:
    """포지션 정보"""
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    size_percent: float         # 시드 대비 비중 (%)
    stop_loss: float
    pyramid_level: PyramidLevel
    take_profit_1: Optional[float] = None   # 30% 익절 목표
    is_active: bool = True
    pnl_percent: float = 0.0


@dataclass
class StrategyState:
    """전략 상태"""
    daily_bias: DailyBias = DailyBias.NEUTRAL
    bias_timestamp: Optional[datetime] = None
    positions: List[Position] = field(default_factory=list)
    total_position_size: float = 0.0
    is_recovery_mode: bool = False
    trade_history: List[Dict[str, Any]] = field(default_factory=list)


class Strategy15M:
    """15분봉 추세 피라미딩 전략"""
    
    # 전략 설정
    INITIAL_SIZE_PERCENT = 10.0      # 초기 진입: 시드의 10%
    PYRAMID_2_SIZE_PERCENT = 20.0    # 2차 피라미딩: 20%
    PYRAMID_3_SIZE_PERCENT = 60.0    # 3차 피라미딩: 60%
    
    PYRAMID_THRESHOLD = 30.0         # 피라미딩 조건: 30% 수익 격차
    TAKE_PROFIT_THRESHOLD = 30.0     # 1차 익절: 30% 수익
    TAKE_PROFIT_SIZE = 35.0          # 익절 물량: 35%
    
    BIAS_CHECK_HOUR = 9              # 바이어스 체크 시간 (오전 9시)
    
    def __init__(self):
        self.signal_detector = SignalDetector(volume_threshold=1.0)
        self.state = StrategyState()
    
    def analyze_daily_bias(self, daily_df: pd.DataFrame) -> DailyBias:
        """
        오전 9시 일봉 캔들 분석으로 데일리 바이어스 설정
        
        Args:
            daily_df: 일봉 OHLCV 데이터
        
        Returns:
            DailyBias (상승/하락/중립)
        """
        if len(daily_df) < 3:
            return DailyBias.NEUTRAL
        
        last_candle = daily_df.iloc[-1]
        prev_candle = daily_df.iloc[-2]
        
        # 시그널 캔들 감지
        signals = self.signal_detector.detect_all_signals(daily_df.tail(5))
        
        # 강한 양봉 + 상승 시그널
        is_bullish_candle = last_candle['close'] > last_candle['open']
        is_bearish_candle = last_candle['close'] < last_candle['open']
        
        bullish_signals = [s for s in signals if s.signal_type in [
            SignalType.BULLISH_ENGULFING, SignalType.MORNING_STAR, 
            SignalType.HAMMER, SignalType.BULLISH_DOJI
        ]]
        
        bearish_signals = [s for s in signals if s.signal_type in [
            SignalType.BEARISH_ENGULFING, SignalType.EVENING_STAR,
            SignalType.INVERTED_HAMMER, SignalType.BEARISH_DOJI
        ]]
        
        # 추세 판단
        if is_bullish_candle and len(bullish_signals) > 0:
            return DailyBias.BULLISH
        elif is_bearish_candle and len(bearish_signals) > 0:
            return DailyBias.BEARISH
        elif is_bullish_candle:
            # 캔들은 양봉이지만 시그널 없음 -> 약한 상승
            return DailyBias.BULLISH
        elif is_bearish_candle:
            return DailyBias.BEARISH
        else:
            return DailyBias.NEUTRAL
    
    def set_daily_bias(self, daily_df: pd.DataFrame, current_time: datetime) -> None:
        """데일리 바이어스 설정 (오전 9시 체크)"""
        # 오전 9시 체크 (이미 오늘 설정했으면 스킵)
        if self.state.bias_timestamp and self.state.bias_timestamp.date() == current_time.date():
            return
        
        if current_time.hour >= self.BIAS_CHECK_HOUR:
            self.state.daily_bias = self.analyze_daily_bias(daily_df)
            self.state.bias_timestamp = current_time
    
    def check_entry_signal(
        self, 
        df_15m: pd.DataFrame, 
        current_price: float,
        current_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        15분봉에서 진입 시그널 체크
        
        Returns:
            진입 신호 딕셔너리 또는 None
        """
        # 바이어스가 중립이면 진입 안함
        if self.state.daily_bias == DailyBias.NEUTRAL:
            return None
        
        # 이미 최대 피라미딩 도달
        if self.state.total_position_size >= 90:  # 10 + 20 + 60
            return None
        
        # 시그널 캔들 감지
        signals = self.signal_detector.detect_all_signals(df_15m.tail(10))
        valid_signals = self.signal_detector.filter_valid_signals(signals)
        
        if not valid_signals:
            return None
        
        latest_signal = valid_signals[-1]
        
        # 바이어스와 시그널 방향 일치 확인
        is_long_signal = latest_signal.signal_type in [
            SignalType.BULLISH_ENGULFING, SignalType.MORNING_STAR,
            SignalType.HAMMER, SignalType.BULLISH_DOJI
        ]
        
        if self.state.daily_bias == DailyBias.BULLISH and is_long_signal:
            direction = "LONG"
        elif self.state.daily_bias == DailyBias.BEARISH and not is_long_signal:
            direction = "SHORT"
        else:
            return None  # 바이어스와 시그널 불일치
        
        # 피라미딩 레벨 결정
        pyramid_level = self._get_next_pyramid_level()
        size_percent = self._get_size_for_level(pyramid_level)
        
        # 피라미딩 조건 확인 (기존 포지션 있을 때)
        if pyramid_level != PyramidLevel.INITIAL:
            if not self._can_pyramid(current_price):
                return None
        
        return {
            'direction': direction,
            'entry_price': latest_signal.entry_price,
            'stop_loss': latest_signal.stop_loss,
            'size_percent': size_percent,
            'pyramid_level': pyramid_level,
            'signal': latest_signal,
            'timestamp': current_time
        }
    
    def _get_next_pyramid_level(self) -> PyramidLevel:
        """다음 피라미딩 레벨 결정"""
        if self.state.total_position_size == 0:
            return PyramidLevel.INITIAL
        elif self.state.total_position_size <= 10:
            return PyramidLevel.LEVEL_2
        elif self.state.total_position_size <= 30:
            return PyramidLevel.LEVEL_3
        else:
            return PyramidLevel.LEVEL_3
    
    def _get_size_for_level(self, level: PyramidLevel) -> float:
        """피라미딩 레벨에 따른 비중"""
        sizes = {
            PyramidLevel.INITIAL: self.INITIAL_SIZE_PERCENT,
            PyramidLevel.LEVEL_2: self.PYRAMID_2_SIZE_PERCENT,
            PyramidLevel.LEVEL_3: self.PYRAMID_3_SIZE_PERCENT
        }
        return sizes[level]
    
    def _can_pyramid(self, current_price: float) -> bool:
        """피라미딩 가능 여부 (30% 수익 격차)"""
        if not self.state.positions:
            return True
        
        last_position = self.state.positions[-1]
        
        if last_position.direction == "LONG":
            pnl = (current_price - last_position.entry_price) / last_position.entry_price * 100
        else:
            pnl = (last_position.entry_price - current_price) / last_position.entry_price * 100
        
        return pnl >= self.PYRAMID_THRESHOLD
    
    def enter_position(self, entry_info: Dict[str, Any]) -> Position:
        """포지션 진입"""
        position = Position(
            direction=entry_info['direction'],
            entry_price=entry_info['entry_price'],
            entry_time=entry_info['timestamp'],
            size_percent=entry_info['size_percent'],
            stop_loss=entry_info['stop_loss'],
            pyramid_level=entry_info['pyramid_level'],
            take_profit_1=self._calculate_take_profit(
                entry_info['entry_price'], 
                entry_info['direction']
            )
        )
        
        self.state.positions.append(position)
        self.state.total_position_size += position.size_percent
        
        return position
    
    def _calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """30% 익절 목표가 계산"""
        if direction == "LONG":
            return entry_price * (1 + self.TAKE_PROFIT_THRESHOLD / 100)
        else:
            return entry_price * (1 - self.TAKE_PROFIT_THRESHOLD / 100)
    
    def update_positions(self, current_price: float, current_time: datetime) -> List[Dict[str, Any]]:
        """
        포지션 업데이트 (손절/익절 체크)
        
        Returns:
            발생한 이벤트 리스트
        """
        events = []
        
        for position in self.state.positions:
            if not position.is_active:
                continue
            
            # PnL 업데이트
            if position.direction == "LONG":
                position.pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
            else:
                position.pnl_percent = (position.entry_price - current_price) / position.entry_price * 100
            
            # 손절 체크
            if self._is_stop_loss_hit(position, current_price):
                events.append(self._close_position(position, current_price, current_time, "STOP_LOSS"))
                continue
            
            # 익절 체크 (30% 달성 시 35% 물량)
            if position.pnl_percent >= self.TAKE_PROFIT_THRESHOLD:
                events.append({
                    'type': 'TAKE_PROFIT',
                    'position': position,
                    'pnl_percent': position.pnl_percent,
                    'exit_percent': self.TAKE_PROFIT_SIZE,
                    'timestamp': current_time,
                    'message': f"1차 익절: {position.pnl_percent:.1f}% 수익, {self.TAKE_PROFIT_SIZE}% 물량 청산"
                })
                # 본절로 스탑로스 이동
                position.stop_loss = position.entry_price
        
        return events
    
    def _is_stop_loss_hit(self, position: Position, current_price: float) -> bool:
        """손절가 도달 여부"""
        if position.direction == "LONG":
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _close_position(
        self, 
        position: Position, 
        exit_price: float, 
        exit_time: datetime,
        reason: str
    ) -> Dict[str, Any]:
        """포지션 청산"""
        position.is_active = False
        self.state.total_position_size -= position.size_percent
        
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - exit_price) / position.entry_price * 100
        
        trade = {
            'type': reason,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'size_percent': position.size_percent,
            'pnl_percent': pnl,
            'message': f"{'손절' if reason == 'STOP_LOSS' else '익절'}: {pnl:.1f}%"
        }
        
        self.state.trade_history.append(trade)
        
        # 손절 시 복구 모드 활성화
        if reason == 'STOP_LOSS':
            self.state.is_recovery_mode = True
        
        return trade
    
    def get_summary(self) -> Dict[str, Any]:
        """전략 요약 정보"""
        active_positions = [p for p in self.state.positions if p.is_active]
        total_pnl = sum(t['pnl_percent'] * t['size_percent'] / 100 for t in self.state.trade_history)
        
        return {
            'strategy': '15분봉 추세 피라미딩',
            'daily_bias': self.state.daily_bias.value,
            'active_positions': len(active_positions),
            'total_position_size': self.state.total_position_size,
            'is_recovery_mode': self.state.is_recovery_mode,
            'total_trades': len(self.state.trade_history),
            'total_pnl_percent': total_pnl
        }
