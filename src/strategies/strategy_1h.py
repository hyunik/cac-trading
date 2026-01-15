"""
전략 B: 1시간봉 볼린저밴드/캔들 매매법 (타이트한 역추세)

- 진입 조건: 1시간봉 볼린저밴드 상/하단선 터치 시 시그널 캔들 발생 여부 체크
- 진입 비중: 시드의 25% 고정
- 손절 설정: 진입과 동시에 시그널 캔들의 꼬리 끝에 세팅
- 익절 구간:
  - 1차: 0.8%(레버리지 미포함) 수익 시 35% 익절
  - 2차: 볼밴 반대편 터치 시 35% 익절
  - 3차: 거래량 폭증 시 전량 익절
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd

from ..analysis.signal_detector import SignalDetector, CandleSignal, SignalType
from ..analysis.bollinger_bands import BollingerBands, BandPosition


class TakeProfitStage(Enum):
    """익절 단계"""
    STAGE_1 = 1   # 0.8% 수익
    STAGE_2 = 2   # 볼밴 반대편 터치
    STAGE_3 = 3   # 거래량 폭증


@dataclass
class Position1H:
    """1시간봉 전략 포지션"""
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    size_percent: float         # 시드 대비 비중 (%)
    stop_loss: float            # 시그널 캔들 꼬리 끝
    remaining_size: float       # 남은 비중 (익절 후 감소)
    take_profit_stage: int = 0  # 현재 익절 단계 (0: 미익절)
    is_active: bool = True
    pnl_percent: float = 0.0
    entry_band_position: Optional[BandPosition] = None


@dataclass
class Strategy1HState:
    """1시간봉 전략 상태"""
    position: Optional[Position1H] = None
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    is_recovery_mode: bool = False
    last_stop_loss_direction: Optional[str] = None  # 마지막 손절 방향 (복구용)


class Strategy1H:
    """1시간봉 볼린저밴드 역추세 전략"""
    
    # 전략 설정
    POSITION_SIZE_PERCENT = 25.0    # 고정 진입 비중: 25%
    
    # 익절 설정
    TAKE_PROFIT_1_PERCENT = 0.8     # 1차 익절: 0.8%
    TAKE_PROFIT_1_SIZE = 35.0       # 1차 익절 물량: 35%
    TAKE_PROFIT_2_SIZE = 35.0       # 2차 익절 물량: 35%
    TAKE_PROFIT_3_SIZE = 30.0       # 3차 익절 물량: 나머지 전량
    
    VOLUME_SPIKE_RATIO = 3.0        # 거래량 폭증 기준: 평균 대비 3배
    
    def __init__(self):
        self.signal_detector = SignalDetector(volume_threshold=1.0)
        self.bollinger = BollingerBands(period=20, std_dev=2.0, touch_threshold=0.02)
        self.state = Strategy1HState()
    
    def check_entry_signal(
        self, 
        df_1h: pd.DataFrame,
        current_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        1시간봉에서 진입 시그널 체크
        
        Returns:
            진입 신호 딕셔너리 또는 None
        """
        # 이미 포지션이 있으면 진입 안함
        if self.state.position is not None and self.state.position.is_active:
            return None
        
        # 볼린저밴드 터치 확인
        is_touch, direction, bb_desc = self.bollinger.get_band_touch_signal(df_1h)
        
        if not is_touch:
            return None
        
        # 시그널 캔들 감지
        signals = self.signal_detector.detect_all_signals(df_1h.tail(10))
        valid_signals = self.signal_detector.filter_valid_signals(signals)
        
        if not valid_signals:
            return None
        
        latest_signal = valid_signals[-1]
        
        # 볼밴 터치 방향과 시그널 방향 일치 확인
        long_signals = [SignalType.BULLISH_ENGULFING, SignalType.MORNING_STAR, 
                        SignalType.HAMMER, SignalType.BULLISH_DOJI]
        short_signals = [SignalType.BEARISH_ENGULFING, SignalType.EVENING_STAR,
                         SignalType.INVERTED_HAMMER, SignalType.BEARISH_DOJI]
        
        if direction == "LONG" and latest_signal.signal_type in long_signals:
            pass  # OK
        elif direction == "SHORT" and latest_signal.signal_type in short_signals:
            pass  # OK
        else:
            return None  # 방향 불일치
        
        band_data = self.bollinger.get_current_band(df_1h)
        
        return {
            'direction': direction,
            'entry_price': latest_signal.entry_price,
            'stop_loss': latest_signal.stop_loss,
            'size_percent': self.POSITION_SIZE_PERCENT,
            'signal': latest_signal,
            'band_position': band_data.position if band_data else None,
            'band_description': bb_desc,
            'timestamp': current_time
        }
    
    def enter_position(self, entry_info: Dict[str, Any]) -> Position1H:
        """포지션 진입"""
        position = Position1H(
            direction=entry_info['direction'],
            entry_price=entry_info['entry_price'],
            entry_time=entry_info['timestamp'],
            size_percent=entry_info['size_percent'],
            stop_loss=entry_info['stop_loss'],
            remaining_size=entry_info['size_percent'],
            entry_band_position=entry_info.get('band_position')
        )
        
        self.state.position = position
        
        return position
    
    def update_position(
        self, 
        df_1h: pd.DataFrame,
        current_price: float, 
        current_volume: float,
        current_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        포지션 업데이트 (손절/익절 체크)
        
        Returns:
            발생한 이벤트 리스트
        """
        events = []
        position = self.state.position
        
        if position is None or not position.is_active:
            return events
        
        # PnL 업데이트
        if position.direction == "LONG":
            position.pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
        else:
            position.pnl_percent = (position.entry_price - current_price) / position.entry_price * 100
        
        # 1. 손절 체크
        if self._is_stop_loss_hit(position, current_price):
            event = self._close_position(position, current_price, current_time, "STOP_LOSS", 100.0)
            events.append(event)
            return events
        
        # 2. 1차 익절: 0.8% 수익
        if position.take_profit_stage < 1 and position.pnl_percent >= self.TAKE_PROFIT_1_PERCENT:
            event = self._partial_exit(
                position, current_price, current_time, 
                TakeProfitStage.STAGE_1, self.TAKE_PROFIT_1_SIZE
            )
            events.append(event)
        
        # 3. 2차 익절: 볼밴 반대편 터치
        if position.take_profit_stage < 2:
            is_opposite_touch = self._check_opposite_band_touch(df_1h, position.direction)
            if is_opposite_touch:
                event = self._partial_exit(
                    position, current_price, current_time,
                    TakeProfitStage.STAGE_2, self.TAKE_PROFIT_2_SIZE
                )
                events.append(event)
        
        # 4. 3차 익절: 거래량 폭증
        if position.take_profit_stage < 3:
            volume_ma = df_1h['volume'].rolling(window=20).mean().iloc[-1]
            if pd.notna(volume_ma) and current_volume >= volume_ma * self.VOLUME_SPIKE_RATIO:
                event = self._close_position(
                    position, current_price, current_time, 
                    "TAKE_PROFIT_3", position.remaining_size
                )
                events.append(event)
        
        return events
    
    def _is_stop_loss_hit(self, position: Position1H, current_price: float) -> bool:
        """손절가 도달 여부"""
        if position.direction == "LONG":
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _check_opposite_band_touch(self, df_1h: pd.DataFrame, entry_direction: str) -> bool:
        """볼밴 반대편 터치 확인"""
        if entry_direction == "LONG":
            # 롱 진입 = 하단 터치 진입 -> 상단 터치 확인
            return self.bollinger.is_touching_upper(df_1h)
        else:
            # 숏 진입 = 상단 터치 진입 -> 하단 터치 확인
            return self.bollinger.is_touching_lower(df_1h)
    
    def _partial_exit(
        self,
        position: Position1H,
        exit_price: float,
        exit_time: datetime,
        stage: TakeProfitStage,
        exit_percent: float
    ) -> Dict[str, Any]:
        """부분 익절"""
        actual_exit_size = min(exit_percent, position.remaining_size)
        position.remaining_size -= actual_exit_size
        position.take_profit_stage = stage.value
        
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - exit_price) / position.entry_price * 100
        
        stage_names = {
            TakeProfitStage.STAGE_1: "1차 익절 (0.8% 수익)",
            TakeProfitStage.STAGE_2: "2차 익절 (볼밴 반대편 터치)",
            TakeProfitStage.STAGE_3: "3차 익절 (거래량 폭증)"
        }
        
        event = {
            'type': f'TAKE_PROFIT_{stage.value}',
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_size_percent': actual_exit_size,
            'remaining_size_percent': position.remaining_size,
            'pnl_percent': pnl,
            'message': f"{stage_names[stage]}: {pnl:.2f}%, {actual_exit_size:.0f}% 물량 청산"
        }
        
        self.state.trade_history.append(event)
        
        # 남은 물량이 없으면 포지션 종료
        if position.remaining_size <= 0:
            position.is_active = False
        
        return event
    
    def _close_position(
        self,
        position: Position1H,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        exit_percent: float
    ) -> Dict[str, Any]:
        """포지션 청산"""
        position.is_active = False
        
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - exit_price) / position.entry_price * 100
        
        event = {
            'type': reason,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'size_percent': exit_percent,
            'pnl_percent': pnl,
            'message': f"{'손절' if 'STOP' in reason else '익절'}: {pnl:.2f}%"
        }
        
        self.state.trade_history.append(event)
        
        # 손절 시 복구 모드 활성화
        if reason == 'STOP_LOSS':
            self.state.is_recovery_mode = True
            self.state.last_stop_loss_direction = position.direction
        
        return event
    
    def get_summary(self) -> Dict[str, Any]:
        """전략 요약 정보"""
        has_position = self.state.position is not None and self.state.position.is_active
        
        total_pnl = 0.0
        for trade in self.state.trade_history:
            size = trade.get('size_percent', trade.get('exit_size_percent', 0))
            total_pnl += trade['pnl_percent'] * size / 100
        
        return {
            'strategy': '1시간봉 볼밴 역추세',
            'has_position': has_position,
            'current_pnl': self.state.position.pnl_percent if has_position else 0,
            'remaining_size': self.state.position.remaining_size if has_position else 0,
            'is_recovery_mode': self.state.is_recovery_mode,
            'total_trades': len(self.state.trade_history),
            'total_pnl_percent': total_pnl
        }
