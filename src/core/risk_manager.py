"""
리스크 관리 모듈
- 최대 포지션 크기 제한
- 일일/주간 손실 한도
- 연속 손실 제한
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """리스크 한도 설정"""
    max_position_size: float = 90.0       # 최대 포지션 비중 (%)
    max_daily_loss: float = 10.0          # 일일 최대 손실 (%)
    max_weekly_loss: float = 20.0         # 주간 최대 손실 (%)
    max_consecutive_losses: int = 3       # 최대 연속 손실 횟수
    max_single_trade_risk: float = 6.0    # 단일 거래 최대 리스크 (%)


class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Args:
            limits: 리스크 한도 설정
        """
        self.limits = limits or RiskLimits()
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._current_position_size: float = 0.0
        self._last_reset_date: datetime = datetime.now().date()
        self._last_week_reset: datetime = self._get_week_start()
        self._trade_log: List[Dict] = []
        self._is_trading_allowed: bool = True
        self._block_reason: Optional[str] = None
    
    def _get_week_start(self) -> datetime:
        """이번 주 월요일 반환"""
        today = datetime.now()
        return today - timedelta(days=today.weekday())
    
    def _check_daily_reset(self) -> None:
        """일일 리셋 체크"""
        today = datetime.now().date()
        if today > self._last_reset_date:
            self._daily_pnl = 0.0
            self._last_reset_date = today
    
    def _check_weekly_reset(self) -> None:
        """주간 리셋 체크"""
        week_start = self._get_week_start()
        if week_start > self._last_week_reset:
            self._weekly_pnl = 0.0
            self._last_week_reset = week_start
            self._consecutive_losses = 0
            self._is_trading_allowed = True
            self._block_reason = None
    
    def check_entry_allowed(
        self, 
        position_size: float,
        stop_loss_percent: float
    ) -> tuple[bool, str]:
        """
        진입 허용 여부 체크
        
        Args:
            position_size: 진입할 포지션 크기 (%)
            stop_loss_percent: 손절 비율 (%)
        
        Returns:
            (허용 여부, 사유)
        """
        self._check_daily_reset()
        self._check_weekly_reset()
        
        # 거래 차단 상태
        if not self._is_trading_allowed:
            return (False, f"거래 차단: {self._block_reason}")
        
        # 최대 포지션 크기
        total_size = self._current_position_size + position_size
        if total_size > self.limits.max_position_size:
            return (
                False, 
                f"최대 포지션 초과: 현재 {self._current_position_size:.1f}% + 신규 {position_size:.1f}% > 한도 {self.limits.max_position_size:.1f}%"
            )
        
        # 단일 거래 리스크
        if stop_loss_percent > self.limits.max_single_trade_risk:
            return (
                False,
                f"손절폭 초과: {stop_loss_percent:.1f}% > 한도 {self.limits.max_single_trade_risk:.1f}%"
            )
        
        # 일일 손실 한도 (이미 손실 많으면 진입 제한)
        if self._daily_pnl <= -self.limits.max_daily_loss * 0.8:  # 80% 도달 시 경고
            return (
                False,
                f"일일 손실 한도 근접: {self._daily_pnl:.1f}% (한도: {self.limits.max_daily_loss:.1f}%)"
            )
        
        # 연속 손실
        if self._consecutive_losses >= self.limits.max_consecutive_losses:
            return (
                False,
                f"연속 손실 한도: {self._consecutive_losses}회 연속 손실"
            )
        
        return (True, "진입 허용")
    
    def record_trade_result(self, pnl_percent: float, size_percent: float) -> None:
        """
        거래 결과 기록
        
        Args:
            pnl_percent: 손익률 (%)
            size_percent: 거래 비중 (%)
        """
        self._check_daily_reset()
        self._check_weekly_reset()
        
        # 가중 PnL
        weighted_pnl = pnl_percent * size_percent / 100
        self._daily_pnl += weighted_pnl
        self._weekly_pnl += weighted_pnl
        
        # 연속 손실 추적
        if pnl_percent < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # 거래 로그
        self._trade_log.append({
            'timestamp': datetime.now(),
            'pnl_percent': pnl_percent,
            'size_percent': size_percent,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl
        })
        
        # 한도 체크 및 차단
        self._check_limits()
    
    def _check_limits(self) -> None:
        """한도 체크 및 거래 차단"""
        # 일일 손실 한도
        if self._daily_pnl <= -self.limits.max_daily_loss:
            self._is_trading_allowed = False
            self._block_reason = f"일일 손실 한도 도달: {self._daily_pnl:.1f}%"
        
        # 주간 손실 한도
        if self._weekly_pnl <= -self.limits.max_weekly_loss:
            self._is_trading_allowed = False
            self._block_reason = f"주간 손실 한도 도달: {self._weekly_pnl:.1f}%"
        
        # 연속 손실
        if self._consecutive_losses >= self.limits.max_consecutive_losses:
            self._is_trading_allowed = False
            self._block_reason = f"연속 {self._consecutive_losses}회 손실"
    
    def update_position_size(self, size_change: float) -> None:
        """포지션 크기 업데이트"""
        self._current_position_size += size_change
        self._current_position_size = max(0, self._current_position_size)
    
    def get_risk_level(self) -> RiskLevel:
        """현재 리스크 레벨"""
        if not self._is_trading_allowed:
            return RiskLevel.CRITICAL
        
        # 일일 손실 기준
        daily_ratio = abs(self._daily_pnl) / self.limits.max_daily_loss if self._daily_pnl < 0 else 0
        
        if daily_ratio >= 0.8:
            return RiskLevel.HIGH
        elif daily_ratio >= 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def force_unlock(self) -> None:
        """강제 잠금 해제 (관리자용)"""
        self._is_trading_allowed = True
        self._block_reason = None
        self._consecutive_losses = 0
    
    def get_status(self) -> Dict[str, Any]:
        """현재 리스크 상태"""
        self._check_daily_reset()
        self._check_weekly_reset()
        
        return {
            'is_trading_allowed': self._is_trading_allowed,
            'block_reason': self._block_reason,
            'risk_level': self.get_risk_level().value,
            'current_position_size': self._current_position_size,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'consecutive_losses': self._consecutive_losses,
            'limits': {
                'max_position': self.limits.max_position_size,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_weekly_loss': self.limits.max_weekly_loss,
                'max_consecutive_losses': self.limits.max_consecutive_losses
            }
        }
