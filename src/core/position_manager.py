"""
포지션 관리 모듈
- 전체 포지션 추적
- 손익 계산
- 거래 기록 관리
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import json


class PositionStatus(Enum):
    """포지션 상태"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Trade:
    """거래 기록"""
    id: str
    strategy: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    size_percent: float = 0.0
    pnl_percent: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy': self.strategy,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'size_percent': self.size_percent,
            'pnl_percent': self.pnl_percent,
            'status': self.status.value,
            'metadata': self.metadata
        }


class PositionManager:
    """포지션 관리자"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Args:
            initial_capital: 초기 자본 (USDT)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self._trade_counter = 0
    
    def create_trade(
        self,
        strategy: str,
        symbol: str,
        direction: str,
        entry_price: float,
        size_percent: float,
        metadata: Optional[Dict] = None
    ) -> Trade:
        """새 거래 생성"""
        self._trade_counter += 1
        trade_id = f"{strategy[:3].upper()}-{self._trade_counter:04d}"
        
        trade = Trade(
            id=trade_id,
            strategy=strategy,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size_percent=size_percent,
            metadata=metadata or {}
        )
        
        self.trades.append(trade)
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        partial_percent: Optional[float] = None
    ) -> Optional[Trade]:
        """거래 청산"""
        trade = self.get_trade(trade_id)
        if not trade:
            return None
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        
        # PnL 계산
        if trade.direction == "LONG":
            trade.pnl_percent = (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            trade.pnl_percent = (trade.entry_price - exit_price) / trade.entry_price * 100
        
        # 부분 청산
        if partial_percent and partial_percent < 100:
            trade.status = PositionStatus.PARTIAL
            # 새로운 거래로 잔여분 생성
            remaining = trade.size_percent * (100 - partial_percent) / 100
            if remaining > 0:
                self.create_trade(
                    strategy=trade.strategy,
                    symbol=trade.symbol,
                    direction=trade.direction,
                    entry_price=trade.entry_price,
                    size_percent=remaining,
                    metadata={**trade.metadata, 'parent_trade': trade_id}
                )
            trade.size_percent = trade.size_percent * partial_percent / 100
        else:
            trade.status = PositionStatus.CLOSED
        
        # 자본 업데이트
        pnl_amount = self.initial_capital * (trade.size_percent / 100) * (trade.pnl_percent / 100)
        self.current_capital += pnl_amount
        
        return trade
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """거래 조회"""
        for trade in self.trades:
            if trade.id == trade_id:
                return trade
        return None
    
    def get_open_trades(self, strategy: Optional[str] = None) -> List[Trade]:
        """활성 거래 조회"""
        open_trades = [t for t in self.trades if t.status == PositionStatus.OPEN]
        if strategy:
            open_trades = [t for t in open_trades if t.strategy == strategy]
        return open_trades
    
    def get_closed_trades(
        self, 
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Trade]:
        """청산된 거래 조회"""
        closed = [t for t in self.trades if t.status in [PositionStatus.CLOSED, PositionStatus.PARTIAL]]
        
        if strategy:
            closed = [t for t in closed if t.strategy == strategy]
        if start_date:
            closed = [t for t in closed if t.exit_time and t.exit_time >= start_date]
        if end_date:
            closed = [t for t in closed if t.exit_time and t.exit_time <= end_date]
        
        return closed
    
    def get_total_pnl(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """총 손익 계산"""
        closed = self.get_closed_trades(strategy)
        
        total_pnl_percent = 0.0
        total_pnl_amount = 0.0
        
        for trade in closed:
            weighted_pnl = trade.pnl_percent * trade.size_percent / 100
            total_pnl_percent += weighted_pnl
            total_pnl_amount += self.initial_capital * weighted_pnl / 100
        
        return {
            'pnl_percent': total_pnl_percent,
            'pnl_amount': total_pnl_amount,
            'current_capital': self.current_capital,
            'roi': (self.current_capital - self.initial_capital) / self.initial_capital * 100
        }
    
    def get_statistics(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """거래 통계"""
        closed = self.get_closed_trades(strategy)
        
        if not closed:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }
        
        wins = [t for t in closed if t.pnl_percent > 0]
        losses = [t for t in closed if t.pnl_percent <= 0]
        
        total_trades = len(closed)
        win_rate = len(wins) / total_trades * 100
        
        avg_win = sum(t.pnl_percent for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_percent for t in losses) / len(losses) if losses else 0
        
        gross_profit = sum(t.pnl_percent * t.size_percent for t in wins)
        gross_loss = abs(sum(t.pnl_percent * t.size_percent for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': max((t.pnl_percent for t in closed), default=0),
            'worst_trade': min((t.pnl_percent for t in closed), default=0)
        }
    
    def export_trades(self, filepath: str) -> None:
        """거래 기록 내보내기"""
        data = [t.to_dict() for t in self.trades]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 요약"""
        pnl = self.get_total_pnl()
        stats = self.get_statistics()
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': pnl,
            'statistics': stats,
            'open_positions': len(self.get_open_trades())
        }
