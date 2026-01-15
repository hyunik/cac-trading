"""
Discord 알림 모듈
- 실시간 매매 알림 (진입/익절/손절)
- 주간 리포트 (매주 일요일)
- AI 전략 비교 분석 및 개선점 제안
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Discord Webhook 기반 알림 시스템"""
    
    # 이모지 매핑
    EMOJI = {
        'entry_long': '🟢',
        'entry_short': '🔴',
        'take_profit': '🟡',
        'stop_loss': '🔴',
        'weekly_report': '📊',
        'warning': '⚠️',
        'success': '✅',
        'info': 'ℹ️'
    }
    
    def __init__(self, webhook_url: str, llm_client: Optional[Any] = None):
        """
        Args:
            webhook_url: Discord Webhook URL
            llm_client: LLM 클라이언트 (주간 분석용)
        """
        self.webhook_url = webhook_url
        self.llm_client = llm_client
        self._trade_log: List[Dict[str, Any]] = []
    
    async def send_message(self, content: str, embed: Optional[Dict] = None) -> bool:
        """
        Discord 메시지 전송
        
        Args:
            content: 메시지 내용
            embed: 임베드 메시지 (옵션)
        
        Returns:
            전송 성공 여부
        """
        payload = {"content": content}
        
        if embed:
            payload["embeds"] = [embed]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status in [200, 204]:
                        return True
                    else:
                        logger.error(f"Discord API error: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False
    
    async def notify_entry(
        self,
        strategy_name: str,
        direction: str,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        size_percent: float,
        signal_description: str,
        timestamp: datetime
    ) -> bool:
        """진입 알림"""
        emoji = self.EMOJI['entry_long'] if direction == "LONG" else self.EMOJI['entry_short']
        
        embed = {
            "title": f"{emoji} 진입 알림 | {strategy_name}",
            "color": 0x00ff00 if direction == "LONG" else 0xff0000,
            "fields": [
                {"name": "심볼", "value": symbol, "inline": True},
                {"name": "방향", "value": direction, "inline": True},
                {"name": "비중", "value": f"{size_percent:.1f}%", "inline": True},
                {"name": "진입가", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "손절가", "value": f"${stop_loss:,.2f}", "inline": True},
                {"name": "손절폭", "value": f"{abs(entry_price - stop_loss) / entry_price * 100:.2f}%", "inline": True},
                {"name": "시그널", "value": signal_description, "inline": False}
            ],
            "timestamp": timestamp.isoformat(),
            "footer": {"text": "CAC Trading System"}
        }
        
        # 거래 로그 저장
        self._trade_log.append({
            'type': 'ENTRY',
            'strategy': strategy_name,
            'direction': direction,
            'symbol': symbol,
            'price': entry_price,
            'size_percent': size_percent,
            'timestamp': timestamp
        })
        
        return await self.send_message("", embed)
    
    async def notify_take_profit(
        self,
        strategy_name: str,
        direction: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_percent: float,
        exit_size_percent: float,
        stage: str,
        timestamp: datetime
    ) -> bool:
        """익절 알림"""
        embed = {
            "title": f"{self.EMOJI['take_profit']} 익절 알림 | {strategy_name}",
            "color": 0xffd700,  # 골드
            "fields": [
                {"name": "심볼", "value": symbol, "inline": True},
                {"name": "방향", "value": direction, "inline": True},
                {"name": "단계", "value": stage, "inline": True},
                {"name": "진입가", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "청산가", "value": f"${exit_price:,.2f}", "inline": True},
                {"name": "수익률", "value": f"+{pnl_percent:.2f}%", "inline": True},
                {"name": "청산 물량", "value": f"{exit_size_percent:.0f}%", "inline": True}
            ],
            "timestamp": timestamp.isoformat(),
            "footer": {"text": "CAC Trading System"}
        }
        
        self._trade_log.append({
            'type': 'TAKE_PROFIT',
            'strategy': strategy_name,
            'direction': direction,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_percent,
            'size_percent': exit_size_percent,
            'timestamp': timestamp
        })
        
        return await self.send_message("", embed)
    
    async def notify_stop_loss(
        self,
        strategy_name: str,
        direction: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_percent: float,
        size_percent: float,
        timestamp: datetime
    ) -> bool:
        """손절 알림"""
        embed = {
            "title": f"{self.EMOJI['stop_loss']} 손절 알림 | {strategy_name}",
            "color": 0xff0000,  # 빨강
            "fields": [
                {"name": "심볼", "value": symbol, "inline": True},
                {"name": "방향", "value": direction, "inline": True},
                {"name": "손실률", "value": f"{pnl_percent:.2f}%", "inline": True},
                {"name": "진입가", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "청산가", "value": f"${exit_price:,.2f}", "inline": True},
                {"name": "청산 물량", "value": f"{size_percent:.0f}%", "inline": True}
            ],
            "timestamp": timestamp.isoformat(),
            "footer": {"text": "CAC Trading System"}
        }
        
        self._trade_log.append({
            'type': 'STOP_LOSS',
            'strategy': strategy_name,
            'direction': direction,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_percent,
            'size_percent': size_percent,
            'timestamp': timestamp
        })
        
        return await self.send_message("", embed)
    
    async def send_weekly_report(
        self,
        strategy_15m_summary: Dict[str, Any],
        strategy_1h_summary: Dict[str, Any],
        week_start: datetime,
        week_end: datetime
    ) -> bool:
        """
        주간 리포트 전송 (매주 일요일 21:00)
        
        두 전략의 주간 성과를 비교하고 AI가 개선점 제안
        """
        # 주간 거래 필터링
        week_trades = [
            t for t in self._trade_log
            if week_start <= t['timestamp'] <= week_end
        ]
        
        # 전략별 통계 계산
        stats_15m = self._calculate_strategy_stats(week_trades, "15분봉 추세 피라미딩")
        stats_1h = self._calculate_strategy_stats(week_trades, "1시간봉 볼밴 역추세")
        
        # 비교 임베드 생성
        comparison_embed = {
            "title": f"{self.EMOJI['weekly_report']} 주간 매매 리포트",
            "description": f"📅 {week_start.strftime('%Y-%m-%d')} ~ {week_end.strftime('%Y-%m-%d')}",
            "color": 0x3498db,
            "fields": [
                # 15분봉 전략
                {"name": "━━━ 전략 A: 15분봉 ━━━", "value": "\u200b", "inline": False},
                {"name": "총 거래", "value": f"{stats_15m['total_trades']}건", "inline": True},
                {"name": "승률", "value": f"{stats_15m['win_rate']:.1f}%", "inline": True},
                {"name": "총 수익률", "value": f"{stats_15m['total_pnl']:+.2f}%", "inline": True},
                {"name": "평균 수익", "value": f"{stats_15m['avg_win']:+.2f}%", "inline": True},
                {"name": "평균 손실", "value": f"{stats_15m['avg_loss']:.2f}%", "inline": True},
                {"name": "최대 손실", "value": f"{stats_15m['max_loss']:.2f}%", "inline": True},
                
                # 1시간봉 전략
                {"name": "━━━ 전략 B: 1시간봉 ━━━", "value": "\u200b", "inline": False},
                {"name": "총 거래", "value": f"{stats_1h['total_trades']}건", "inline": True},
                {"name": "승률", "value": f"{stats_1h['win_rate']:.1f}%", "inline": True},
                {"name": "총 수익률", "value": f"{stats_1h['total_pnl']:+.2f}%", "inline": True},
                {"name": "평균 수익", "value": f"{stats_1h['avg_win']:+.2f}%", "inline": True},
                {"name": "평균 손실", "value": f"{stats_1h['avg_loss']:.2f}%", "inline": True},
                {"name": "최대 손실", "value": f"{stats_1h['max_loss']:.2f}%", "inline": True},
            ],
            "footer": {"text": "CAC Trading System - Weekly Report"}
        }
        
        await self.send_message("", comparison_embed)
        
        # AI 개선점 분석
        if self.llm_client:
            ai_analysis = await self._generate_ai_analysis(stats_15m, stats_1h, week_trades)
            
            analysis_embed = {
                "title": "🤖 AI 전략 분석 및 개선점",
                "description": ai_analysis,
                "color": 0x9b59b6,  # 보라색
                "footer": {"text": "Powered by LLM Analysis"}
            }
            
            await self.send_message("", analysis_embed)
        
        # 주간 로그 초기화 (옵션)
        # self._trade_log = []
        
        return True
    
    def _calculate_strategy_stats(
        self, 
        trades: List[Dict[str, Any]], 
        strategy_name: str
    ) -> Dict[str, Any]:
        """전략별 통계 계산"""
        strategy_trades = [t for t in trades if t.get('strategy') == strategy_name]
        
        if not strategy_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_loss': 0.0
            }
        
        # 수익/손실 거래 분리
        exit_trades = [t for t in strategy_trades if t['type'] in ['TAKE_PROFIT', 'STOP_LOSS']]
        wins = [t for t in exit_trades if t.get('pnl_percent', 0) > 0]
        losses = [t for t in exit_trades if t.get('pnl_percent', 0) <= 0]
        
        total_trades = len(exit_trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        # 가중 평균 PnL (비중 고려)
        total_pnl = sum(
            t.get('pnl_percent', 0) * t.get('size_percent', 100) / 100 
            for t in exit_trades
        )
        
        avg_win = sum(t.get('pnl_percent', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get('pnl_percent', 0) for t in losses) / len(losses) if losses else 0
        max_loss = min((t.get('pnl_percent', 0) for t in losses), default=0)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_loss': max_loss
        }
    
    async def _generate_ai_analysis(
        self,
        stats_15m: Dict[str, Any],
        stats_1h: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> str:
        """LLM을 사용한 전략 분석 및 개선점 제안"""
        if not self.llm_client:
            return "LLM 클라이언트가 설정되지 않았습니다."
        
        prompt = f"""
다음은 두 가지 암호화폐 선물 매매 전략의 주간 성과입니다:

## 전략 A: 15분봉 추세 피라미딩
- 총 거래: {stats_15m['total_trades']}건
- 승률: {stats_15m['win_rate']:.1f}%
- 총 수익률: {stats_15m['total_pnl']:.2f}%
- 평균 수익: {stats_15m['avg_win']:.2f}%
- 평균 손실: {stats_15m['avg_loss']:.2f}%
- 최대 손실: {stats_15m['max_loss']:.2f}%

## 전략 B: 1시간봉 볼린저밴드 역추세
- 총 거래: {stats_1h['total_trades']}건
- 승률: {stats_1h['win_rate']:.1f}%
- 총 수익률: {stats_1h['total_pnl']:.2f}%
- 평균 수익: {stats_1h['avg_win']:.2f}%
- 평균 손실: {stats_1h['avg_loss']:.2f}%
- 최대 손실: {stats_1h['max_loss']:.2f}%

위 데이터를 분석하여:
1. 어떤 전략이 더 효과적이었는지 평가
2. 각 전략의 강점과 약점 분석
3. 구체적인 개선 방안 3가지 제안

간결하게 한국어로 답변해주세요 (300자 이내).
"""
        
        try:
            # LLM API 호출 (클라이언트 인터페이스에 따라 수정 필요)
            response = await self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return f"AI 분석 생성 실패: {e}"
    
    def schedule_weekly_report(self) -> datetime:
        """다음 주간 리포트 시간 계산 (일요일 21:00)"""
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 21:
            days_until_sunday = 7
        
        next_sunday = now + timedelta(days=days_until_sunday)
        return next_sunday.replace(hour=21, minute=0, second=0, microsecond=0)


# 동기 래퍼 클래스
class DiscordNotifierSync:
    """동기 버전 Discord 알림"""
    
    def __init__(self, webhook_url: str, llm_client: Optional[Any] = None):
        self._notifier = DiscordNotifier(webhook_url, llm_client)
    
    def notify_entry(self, **kwargs) -> bool:
        return asyncio.run(self._notifier.notify_entry(**kwargs))
    
    def notify_take_profit(self, **kwargs) -> bool:
        return asyncio.run(self._notifier.notify_take_profit(**kwargs))
    
    def notify_stop_loss(self, **kwargs) -> bool:
        return asyncio.run(self._notifier.notify_stop_loss(**kwargs))
    
    def send_weekly_report(self, **kwargs) -> bool:
        return asyncio.run(self._notifier.send_weekly_report(**kwargs))
