"""
Discord 알림 모듈
- 매일 오전 9시 일일 매매 리포트
- 주간 리포트 (매주 일요일 21:00)
- AI 전략 비교 분석 및 개선점 제안
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Discord Webhook 기반 알림 시스템 (일일/주간 리포트 전용)"""
    
    # 이모지 매핑
    EMOJI = {
        'daily_report': '📋',
        'weekly_report': '📊',
        'profit': '💰',
        'loss': '📉',
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
        """Discord 메시지 전송"""
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
    
    async def send_image(self, image_path: str, content: str = "", embed: Optional[Dict] = None) -> bool:
        """이미지와 함께 Discord 메시지 전송"""
        try:
            import os
            if not os.path.exists(image_path):
                logger.error(f"이미지 파일 없음: {image_path}")
                return False
            
            data = aiohttp.FormData()
            data.add_field('file', open(image_path, 'rb'), 
                          filename=os.path.basename(image_path))
            
            payload = {"content": content}
            if embed:
                import json
                payload["payload_json"] = json.dumps({"embeds": [embed]})
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, data=data) as response:
                    if response.status in [200, 204]:
                        logger.info(f"이미지 전송 성공: {image_path}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Discord 이미지 전송 오류: {response.status} - {error}")
                        return False
        except Exception as e:
            logger.error(f"이미지 전송 오류: {e}")
            return False
    
    async def send_cac_analysis(
        self, 
        symbol: str, 
        chart_path: str, 
        analysis_text: str,
        trend: str = "NEUTRAL",
        signal: str = "HOLD"
    ) -> bool:
        """
        CAC 분석 리포트 전송 (차트 이미지 + 분석)
        
        Args:
            symbol: 코인 심볼
            chart_path: 차트 이미지 경로
            analysis_text: LLM 분석 텍스트
            trend: 추세 (BULLISH/BEARISH/NEUTRAL)
            signal: 신호 (BUY/SELL/HOLD)
        """
        # 색상 결정
        if trend == "BULLISH":
            color = 0x26a69a
            trend_emoji = "📈"
        elif trend == "BEARISH":
            color = 0xef5350
            trend_emoji = "📉"
        else:
            color = 0x9e9e9e
            trend_emoji = "➡️"
        
        # 신호 이모지
        signal_map = {"BUY": "🟢 매수", "SELL": "🔴 매도", "HOLD": "⚪ 관망"}
        signal_text = signal_map.get(signal, "⚪ 관망")
        
        embed = {
            "title": f"{trend_emoji} {symbol} CAC 분석",
            "description": analysis_text[:2000],  # Discord 제한
            "color": color,
            "fields": [
                {"name": "추세", "value": trend, "inline": True},
                {"name": "신호", "value": signal_text, "inline": True}
            ],
            "footer": {"text": "CAC Trading System - Daily Analysis"},
            "timestamp": datetime.now().isoformat()
        }
        
        # 이미지와 함께 전송
        return await self.send_image(chart_path, "", embed)
    
    async def send_daily_cac_report(
        self, 
        analyses: list,
        chart_paths: Dict[str, str]
    ) -> bool:
        """
        전체 코인 일일 CAC 분석 리포트 전송
        
        Args:
            analyses: CACAnalysisResult 리스트
            chart_paths: {symbol: chart_path} 딕셔너리
        """
        # 요약 먼저 전송
        now = datetime.now()
        summary_embed = {
            "title": "📈 일일 CAC 분석 리포트",
            "description": f"📅 {now.strftime('%Y-%m-%d %H:%M')} 기준",
            "color": 0x2196f3,
            "fields": [],
            "footer": {"text": f"총 {len(analyses)}개 코인 분석"}
        }
        
        # 각 코인 요약
        buy_coins = []
        sell_coins = []
        hold_coins = []
        
        for analysis in analyses:
            if analysis.signal == 'BUY':
                buy_coins.append(analysis.symbol)
            elif analysis.signal == 'SELL':
                sell_coins.append(analysis.symbol)
            else:
                hold_coins.append(analysis.symbol)
        
        if buy_coins:
            summary_embed["fields"].append({
                "name": "🟢 매수 신호",
                "value": ", ".join(buy_coins),
                "inline": False
            })
        if sell_coins:
            summary_embed["fields"].append({
                "name": "🔴 매도 신호", 
                "value": ", ".join(sell_coins),
                "inline": False
            })
        if hold_coins:
            summary_embed["fields"].append({
                "name": "⚪ 관망",
                "value": ", ".join(hold_coins),
                "inline": False
            })
        
        await self.send_message("", summary_embed)
        
        # 각 코인별 상세 분석 전송 (매수/매도 신호만)
        for analysis in analyses:
            if analysis.signal in ['BUY', 'SELL']:
                chart_path = chart_paths.get(analysis.symbol)
                if chart_path:
                    await self.send_cac_analysis(
                        symbol=analysis.symbol,
                        chart_path=chart_path,
                        analysis_text=analysis.analysis,
                        trend=analysis.trend,
                        signal=analysis.signal
                    )
                    await asyncio.sleep(1)  # 레이트 리밋 방지
        
        return True
    
    def log_entry(
        self,
        strategy_name: str,
        direction: str,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        size_percent: float,
        signal_description: str,
        timestamp: datetime
    ) -> None:
        """진입 로그 저장 (알림 없음)"""
        self._trade_log.append({
            'type': 'ENTRY',
            'strategy': strategy_name,
            'direction': direction,
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'size_percent': size_percent,
            'signal': signal_description,
            'timestamp': timestamp
        })
        logger.info(f"[LOG] 진입: {symbol} {direction} @ {entry_price:,.4f}")
    
    def log_take_profit(
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
    ) -> None:
        """익절 로그 저장 (알림 없음)"""
        self._trade_log.append({
            'type': 'TAKE_PROFIT',
            'strategy': strategy_name,
            'direction': direction,
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_percent,
            'size_percent': exit_size_percent,
            'stage': stage,
            'timestamp': timestamp
        })
        logger.info(f"[LOG] 익절: {symbol} {pnl_percent:+.2f}%")
    
    def log_stop_loss(
        self,
        strategy_name: str,
        direction: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_percent: float,
        size_percent: float,
        timestamp: datetime
    ) -> None:
        """손절 로그 저장 (알림 없음)"""
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
        logger.info(f"[LOG] 손절: {symbol} {pnl_percent:.2f}%")
    
    # 기존 메서드들은 로그만 저장하도록 변경 (하위 호환성)
    async def notify_entry(self, **kwargs) -> bool:
        """진입 알림 (로그만 저장, Discord 알림 없음)"""
        self.log_entry(**kwargs)
        return True
    
    async def notify_take_profit(self, **kwargs) -> bool:
        """익절 알림 (로그만 저장, Discord 알림 없음)"""
        self.log_take_profit(**kwargs)
        return True
    
    async def notify_stop_loss(self, **kwargs) -> bool:
        """손절 알림 (로그만 저장, Discord 알림 없음)"""
        self.log_stop_loss(**kwargs)
        return True
    
    async def send_daily_report(self) -> bool:
        """
        일일 리포트 전송 (매일 오전 9시)
        전일(어제) 00:00 ~ 23:59 거래 내역 요약
        """
        now = datetime.now()
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday_start.replace(hour=23, minute=59, second=59)
        
        # 전일 거래 필터링
        daily_trades = [
            t for t in self._trade_log
            if yesterday_start <= t['timestamp'] <= yesterday_end
        ]
        
        # 전략별 통계
        stats_15m = self._calculate_strategy_stats(daily_trades, "15분봉 추세 피라미딩")
        stats_1h = self._calculate_strategy_stats(daily_trades, "1시간봉 볼밴 역추세")
        
        # 심볼별 거래 요약
        symbol_summary = self._get_symbol_summary(daily_trades)
        
        # 총 수익률
        total_pnl = stats_15m['total_pnl'] + stats_1h['total_pnl']
        total_trades = stats_15m['total_trades'] + stats_1h['total_trades']
        
        # 임베드 생성
        embed = {
            "title": f"{self.EMOJI['daily_report']} 일일 매매 리포트",
            "description": f"📅 {yesterday_start.strftime('%Y-%m-%d')} (전일)",
            "color": 0x2ecc71 if total_pnl >= 0 else 0xe74c3c,
            "fields": [
                # 요약
                {"name": "📈 총 수익률", "value": f"**{total_pnl:+.2f}%**", "inline": True},
                {"name": "🔢 총 거래", "value": f"{total_trades}건", "inline": True},
                {"name": "\u200b", "value": "\u200b", "inline": True},
                
                # 전략 A
                {"name": "━━ 15분봉 전략 ━━", "value": "\u200b", "inline": False},
                {"name": "거래", "value": f"{stats_15m['total_trades']}건", "inline": True},
                {"name": "승률", "value": f"{stats_15m['win_rate']:.0f}%", "inline": True},
                {"name": "수익률", "value": f"{stats_15m['total_pnl']:+.2f}%", "inline": True},
                
                # 전략 B
                {"name": "━━ 1시간봉 전략 ━━", "value": "\u200b", "inline": False},
                {"name": "거래", "value": f"{stats_1h['total_trades']}건", "inline": True},
                {"name": "승률", "value": f"{stats_1h['win_rate']:.0f}%", "inline": True},
                {"name": "수익률", "value": f"{stats_1h['total_pnl']:+.2f}%", "inline": True},
            ],
            "footer": {"text": "CAC Trading System - Daily Report"},
            "timestamp": now.isoformat()
        }
        
        # 심볼별 요약 추가 (거래가 있는 경우)
        if symbol_summary:
            embed["fields"].append({
                "name": "📊 코인별 거래",
                "value": symbol_summary,
                "inline": False
            })
        
        if total_trades == 0:
            embed["description"] += "\n\n_전일 거래 없음_"
        
        return await self.send_message("", embed)
    
    def _get_symbol_summary(self, trades: List[Dict[str, Any]]) -> str:
        """심볼별 거래 요약 생성"""
        exit_trades = [t for t in trades if t['type'] in ['TAKE_PROFIT', 'STOP_LOSS']]
        
        if not exit_trades:
            return ""
        
        # 심볼별 집계
        symbol_stats = {}
        for trade in exit_trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'pnl': 0.0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade.get('pnl_percent', 0) * trade.get('size_percent', 100) / 100
        
        # 문자열 생성
        lines = []
        for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            emoji = "🟢" if stats['pnl'] >= 0 else "🔴"
            lines.append(f"{emoji} {symbol}: {stats['trades']}건 ({stats['pnl']:+.2f}%)")
        
        return "\n".join(lines[:10])  # 최대 10개
    
    async def send_weekly_report(
        self,
        strategy_15m_summary: Dict[str, Any],
        strategy_1h_summary: Dict[str, Any],
        week_start: datetime,
        week_end: datetime
    ) -> bool:
        """주간 리포트 전송 (매주 일요일 21:00)"""
        # 주간 거래 필터링
        week_trades = [
            t for t in self._trade_log
            if week_start <= t['timestamp'] <= week_end
        ]
        
        # 전략별 통계 계산
        stats_15m = self._calculate_strategy_stats(week_trades, "15분봉 추세 피라미딩")
        stats_1h = self._calculate_strategy_stats(week_trades, "1시간봉 볼밴 역추세")
        
        total_pnl = stats_15m['total_pnl'] + stats_1h['total_pnl']
        
        # 비교 임베드 생성
        comparison_embed = {
            "title": f"{self.EMOJI['weekly_report']} 주간 매매 리포트",
            "description": f"📅 {week_start.strftime('%Y-%m-%d')} ~ {week_end.strftime('%Y-%m-%d')}\n\n**총 주간 수익률: {total_pnl:+.2f}%**",
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
                "color": 0x9b59b6,
                "footer": {"text": "Powered by LLM Analysis"}
            }
            
            await self.send_message("", analysis_embed)
        
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
        
        exit_trades = [t for t in strategy_trades if t['type'] in ['TAKE_PROFIT', 'STOP_LOSS']]
        wins = [t for t in exit_trades if t.get('pnl_percent', 0) > 0]
        losses = [t for t in exit_trades if t.get('pnl_percent', 0) <= 0]
        
        total_trades = len(exit_trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
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
        """LLM을 사용한 전략 분석"""
        if not self.llm_client:
            return "LLM 클라이언트가 설정되지 않았습니다."
        
        prompt = f"""
다음은 두 가지 암호화폐 선물 매매 전략의 주간 성과입니다:

## 전략 A: 15분봉 추세 피라미딩
- 총 거래: {stats_15m['total_trades']}건
- 승률: {stats_15m['win_rate']:.1f}%
- 총 수익률: {stats_15m['total_pnl']:.2f}%

## 전략 B: 1시간봉 볼린저밴드 역추세
- 총 거래: {stats_1h['total_trades']}건
- 승률: {stats_1h['win_rate']:.1f}%
- 총 수익률: {stats_1h['total_pnl']:.2f}%

간결하게 분석하고 개선점 3가지를 제안해주세요 (200자 이내).
"""
        
        try:
            response = await self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return f"AI 분석 생성 실패: {e}"
    
    def schedule_daily_report(self) -> datetime:
        """다음 일일 리포트 시간 계산 (오전 9:00)"""
        now = datetime.now()
        next_report = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now.hour >= 9:
            next_report += timedelta(days=1)
        return next_report
    
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
    
    def send_daily_report(self) -> bool:
        return asyncio.run(self._notifier.send_daily_report())
    
    def send_weekly_report(self, **kwargs) -> bool:
        return asyncio.run(self._notifier.send_weekly_report(**kwargs))
