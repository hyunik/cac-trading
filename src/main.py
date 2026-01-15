"""
AI 하이브리드 코인 선물 페이퍼 트레이딩 시스템
메인 엔트리포인트

15분봉 추세 피라미딩 전략과 1시간봉 볼린저밴드 역추세 전략을 병렬 실행하여 비교
다중 심볼(11개 코인) 동시 트레이딩 지원
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수만 사용

from .data.binance_client import BinanceClient
from .strategies.strategy_15m import Strategy15M
from .strategies.strategy_1h import Strategy1H
from .analysis.fibonacci import FibonacciRetracement
from .core.position_manager import PositionManager
from .core.risk_manager import RiskManager, RiskLimits
from .notifications.discord_notifier import DiscordNotifier


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 트레이딩 대상 코인 목록
TRADING_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT", 
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AAVEUSDT",
    "1000PEPEUSDT",
    "OPUSDT",
    "ARBUSDT",
    "WIFUSDT",
    "JTOUSDT"
]


class SymbolTrader:
    """개별 심볼 트레이더"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.binance = BinanceClient(symbol)
        self.strategy_15m = Strategy15M()
        self.strategy_1h = Strategy1H()
        self.fibonacci = FibonacciRetracement()
        
        # 데이터 저장
        self.df_15m = None
        self.df_1h = None
        self.df_1d = None
    
    async def load_initial_data(self) -> None:
        """초기 데이터 로드"""
        self.df_15m = await self.binance.get_klines('15m', limit=100)
        self.df_1h = await self.binance.get_klines('1h', limit=100)
        self.df_1d = await self.binance.get_klines('1d', limit=30)
        self.strategy_15m.set_daily_bias(self.df_1d, datetime.now())
        logger.info(f"[{self.symbol}] 데이터 로드 완료")
    
    async def close(self) -> None:
        """연결 종료"""
        await self.binance.close()


class MultiSymbolTradingSystem:
    """다중 심볼 트레이딩 시스템"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        discord_webhook: Optional[str] = None,
        initial_capital: float = 10000.0
    ):
        """
        Args:
            symbols: 거래 심볼 리스트
            discord_webhook: Discord Webhook URL
            initial_capital: 초기 자본 (USDT)
        """
        self.symbols = symbols or TRADING_SYMBOLS
        self.traders: Dict[str, SymbolTrader] = {}
        
        # 공통 컴포넌트
        self.position_manager = PositionManager(initial_capital)
        self.risk_manager = RiskManager(RiskLimits())
        self.discord = DiscordNotifier(discord_webhook) if discord_webhook else None
        
        self._running = False
        self._last_weekly_report: Optional[datetime] = None
    
    async def start(self) -> None:
        """시스템 시작"""
        logger.info(f"=== Multi-Symbol Trading System 시작 ===")
        logger.info(f"대상 코인: {len(self.symbols)}개")
        for symbol in self.symbols:
            logger.info(f"  - {symbol}")
        
        self._running = True
        
        # 각 심볼별 트레이더 초기화
        for symbol in self.symbols:
            self.traders[symbol] = SymbolTrader(symbol)
            await self.traders[symbol].load_initial_data()
        
        # 일일/주간 리포트 스케줄러
        asyncio.create_task(self._report_scheduler())
        
        # 각 심볼의 캔들 구독 (병렬)
        tasks = []
        for symbol, trader in self.traders.items():
            task = asyncio.create_task(
                trader.binance.subscribe_klines(
                    intervals=['15m', '1h', '1d'],
                    callback=lambda interval, candle, s=symbol: self._on_candle_update(s, interval, candle)
                )
            )
            tasks.append(task)
        
        # 모든 구독 실행
        await asyncio.gather(*tasks)
    
    async def stop(self) -> None:
        """시스템 종료"""
        logger.info("=== Trading System 종료 ===")
        self._running = False
        
        for trader in self.traders.values():
            await trader.close()
    
    async def _on_candle_update(self, symbol: str, interval: str, candle: Dict[str, Any]) -> None:
        """실시간 캔들 업데이트 콜백"""
        if not self._running:
            return
        
        # 캔들 마감시에만 전략 실행
        if not candle.get('is_closed', False):
            return
        
        trader = self.traders.get(symbol)
        if not trader:
            return
        
        current_price = candle['close']
        current_volume = candle['volume']
        current_time = candle['open_time']
        
        logger.info(f"[{symbol}] 캔들 마감: {interval} @ {current_price:,.4f}")
        
        try:
            if interval == '1d':
                trader.df_1d = await trader.binance.get_klines('1d', limit=30)
                trader.strategy_15m.set_daily_bias(trader.df_1d, current_time)
                logger.info(f"[{symbol}] 바이어스: {trader.strategy_15m.state.daily_bias.value}")
            
            elif interval == '15m':
                trader.df_15m = await trader.binance.get_klines('15m', limit=100)
                await self._execute_strategy_15m(symbol, trader, current_price, current_time)
            
            elif interval == '1h':
                trader.df_1h = await trader.binance.get_klines('1h', limit=100)
                await self._execute_strategy_1h(symbol, trader, current_price, current_volume, current_time)
        
        except Exception as e:
            logger.error(f"[{symbol}] 전략 실행 오류: {e}")
    
    async def _execute_strategy_15m(
        self, 
        symbol: str, 
        trader: SymbolTrader, 
        current_price: float, 
        current_time: datetime
    ) -> None:
        """15분봉 전략 실행"""
        events = trader.strategy_15m.update_positions(current_price, current_time)
        
        for event in events:
            await self._handle_trade_event(symbol, event, "15분봉 추세 피라미딩")
        
        entry_signal = trader.strategy_15m.check_entry_signal(
            trader.df_15m, current_price, current_time
        )
        
        if entry_signal:
            await self._try_entry(symbol, trader, entry_signal, "15분봉 추세 피라미딩")
    
    async def _execute_strategy_1h(
        self,
        symbol: str,
        trader: SymbolTrader,
        current_price: float,
        current_volume: float,
        current_time: datetime
    ) -> None:
        """1시간봉 전략 실행"""
        events = trader.strategy_1h.update_position(
            trader.df_1h, current_price, current_volume, current_time
        )
        
        for event in events:
            await self._handle_trade_event(symbol, event, "1시간봉 볼밴 역추세")
        
        entry_signal = trader.strategy_1h.check_entry_signal(trader.df_1h, current_time)
        
        if entry_signal:
            await self._try_entry(symbol, trader, entry_signal, "1시간봉 볼밴 역추세")
    
    async def _try_entry(
        self, 
        symbol: str, 
        trader: SymbolTrader, 
        entry_info: Dict[str, Any], 
        strategy_name: str
    ) -> None:
        """진입 시도"""
        stop_loss_percent = abs(
            entry_info['entry_price'] - entry_info['stop_loss']
        ) / entry_info['entry_price'] * 100
        
        can_enter, reason = self.risk_manager.check_entry_allowed(
            entry_info['size_percent'],
            stop_loss_percent
        )
        
        if not can_enter:
            logger.warning(f"[{symbol}] 진입 거부: {reason}")
            return
        
        if strategy_name == "15분봉 추세 피라미딩":
            trader.strategy_15m.enter_position(entry_info)
        else:
            trader.strategy_1h.enter_position(entry_info)
        
        self.risk_manager.update_position_size(entry_info['size_percent'])
        
        self.position_manager.create_trade(
            strategy=strategy_name,
            symbol=symbol,
            direction=entry_info['direction'],
            entry_price=entry_info['entry_price'],
            size_percent=entry_info['size_percent']
        )
        
        logger.info(f"[{symbol}] 진입: {strategy_name} {entry_info['direction']} @ {entry_info['entry_price']:,.4f}")
        
        if self.discord:
            signal = entry_info.get('signal')
            await self.discord.notify_entry(
                strategy_name=f"{strategy_name}",
                direction=entry_info['direction'],
                symbol=symbol,
                entry_price=entry_info['entry_price'],
                stop_loss=entry_info['stop_loss'],
                size_percent=entry_info['size_percent'],
                signal_description=signal.description if signal else "시그널 감지",
                timestamp=entry_info['timestamp']
            )
    
    async def _handle_trade_event(self, symbol: str, event: Dict[str, Any], strategy_name: str) -> None:
        """거래 이벤트 처리"""
        event_type = event.get('type', '')
        
        if 'pnl_percent' in event:
            size = event.get('size_percent', event.get('exit_size_percent', 0))
            self.risk_manager.record_trade_result(event['pnl_percent'], size)
            self.risk_manager.update_position_size(-size)
        
        logger.info(f"[{symbol}] 이벤트: {strategy_name} - {event.get('message', event_type)}")
        
        if self.discord:
            if 'STOP_LOSS' in event_type:
                await self.discord.notify_stop_loss(
                    strategy_name=strategy_name,
                    direction=event['direction'],
                    symbol=symbol,
                    entry_price=event['entry_price'],
                    exit_price=event['exit_price'],
                    pnl_percent=event['pnl_percent'],
                    size_percent=event.get('size_percent', 100),
                    timestamp=event.get('exit_time', datetime.now())
                )
            elif 'TAKE_PROFIT' in event_type:
                await self.discord.notify_take_profit(
                    strategy_name=strategy_name,
                    direction=event['direction'],
                    symbol=symbol,
                    entry_price=event['entry_price'],
                    exit_price=event['exit_price'],
                    pnl_percent=event['pnl_percent'],
                    exit_size_percent=event.get('exit_size_percent', event.get('size_percent', 0)),
                    stage=event.get('message', event_type),
                    timestamp=event.get('exit_time', datetime.now())
                )
    
    async def _report_scheduler(self) -> None:
        """일일 리포트(오전 9시) + 주간 리포트(일요일 21시) 스케줄러"""
        last_daily_date = None
        
        while self._running:
            now = datetime.now()
            
            # 일일 리포트: 매일 오전 9시
            if now.hour == 9 and now.minute < 1:
                today = now.date()
                if last_daily_date != today:
                    logger.info("📋 일일 리포트 전송 중...")
                    await self._send_daily_report()
                    last_daily_date = today
            
            # 주간 리포트: 일요일 21시
            if now.weekday() == 6 and now.hour == 21:
                if self._last_weekly_report is None or \
                   (now - self._last_weekly_report).days >= 6:
                    logger.info("📊 주간 리포트 전송 중...")
                    await self._send_weekly_report()
                    self._last_weekly_report = now
            
            await asyncio.sleep(60)
    
    async def _send_daily_report(self) -> None:
        """일일 리포트 전송 (매일 오전 9시)"""
        if not self.discord:
            return
        
        await self.discord.send_daily_report()
        logger.info("일일 리포트 전송 완료")
    
    async def _send_weekly_report(self) -> None:
        """주간 리포트 전송"""
        if not self.discord:
            return
        
        # 전체 전략 요약 집계
        summary_15m = {'strategy': '15분봉 추세 피라미딩', 'total_trades': 0, 'total_pnl_percent': 0}
        summary_1h = {'strategy': '1시간봉 볼밴 역추세', 'total_trades': 0, 'total_pnl_percent': 0}
        
        for trader in self.traders.values():
            s15 = trader.strategy_15m.get_summary()
            s1h = trader.strategy_1h.get_summary()
            summary_15m['total_trades'] += s15.get('total_trades', 0)
            summary_15m['total_pnl_percent'] += s15.get('total_pnl_percent', 0)
            summary_1h['total_trades'] += s1h.get('total_trades', 0)
            summary_1h['total_pnl_percent'] += s1h.get('total_pnl_percent', 0)
        
        week_end = datetime.now()
        week_start = week_end - timedelta(days=7)
        
        await self.discord.send_weekly_report(
            strategy_15m_summary=summary_15m,
            strategy_1h_summary=summary_1h,
            week_start=week_start,
            week_end=week_end
        )
        
        logger.info("주간 리포트 전송 완료")
    
    def get_status(self) -> Dict[str, Any]:
        """시스템 상태"""
        symbol_status = {}
        for symbol, trader in self.traders.items():
            symbol_status[symbol] = {
                'strategy_15m': trader.strategy_15m.get_summary(),
                'strategy_1h': trader.strategy_1h.get_summary()
            }
        
        return {
            'symbols': self.symbols,
            'symbol_count': len(self.symbols),
            'is_running': self._running,
            'symbol_status': symbol_status,
            'risk': self.risk_manager.get_status(),
            'position': self.position_manager.get_summary()
        }


async def main():
    """메인 함수"""
    discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
    
    # 커스텀 심볼 리스트 (환경변수로 오버라이드 가능)
    custom_symbols = os.getenv('TRADING_SYMBOLS')
    symbols = custom_symbols.split(',') if custom_symbols else None
    
    if not discord_webhook:
        logger.warning("DISCORD_WEBHOOK_URL 환경변수가 설정되지 않았습니다.")
    
    system = MultiSymbolTradingSystem(
        symbols=symbols,
        discord_webhook=discord_webhook,
        initial_capital=initial_capital
    )
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
