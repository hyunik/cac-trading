"""
Binance API 클라이언트
- REST API: 과거 캔들 데이터 조회
- WebSocket: 실시간 캔들 데이터 스트리밍
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
import pandas as pd
import aiohttp
import websockets

logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance Futures API 클라이언트"""
    
    BASE_URL = "https://fapi.binance.com"
    WS_URL = "wss://fstream.binance.com/ws"
    
    INTERVALS = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Args:
            symbol: 거래 심볼 (예: BTCUSDT)
        """
        self.symbol = symbol.upper()
        self._ws = None
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._running = False
    
    async def get_klines(
        self, 
        interval: str, 
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        캔들(Kline) 데이터 조회
        
        Args:
            interval: 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d)
            limit: 반환할 캔들 개수 (최대 1500)
            start_time: 시작 시간
            end_time: 종료 시간
        
        Returns:
            OHLCV DataFrame
        """
        endpoint = f"{self.BASE_URL}/fapi/v1/klines"
        
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Binance API error: {response.status} - {error_text}")
                
                data = await response.json()
        
        return self._parse_klines(data)
    
    def _parse_klines(self, data: List) -> pd.DataFrame:
        """Kline 데이터를 DataFrame으로 변환"""
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 타입 변환
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('open_time', inplace=True)
        df.index.name = None
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    async def get_current_price(self) -> float:
        """현재 가격 조회"""
        endpoint = f"{self.BASE_URL}/fapi/v1/ticker/price"
        params = {'symbol': self.symbol}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as response:
                data = await response.json()
                return float(data['price'])
    
    async def subscribe_klines(
        self,
        intervals: List[str],
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        실시간 캔들 데이터 구독
        
        Args:
            intervals: 구독할 간격 리스트 (예: ['15m', '1h'])
            callback: 콜백 함수 (interval, candle_data)
        """
        streams = [f"{self.symbol.lower()}@kline_{interval}" for interval in intervals]
        stream_url = f"{self.WS_URL}/{'/'.join(streams)}"
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(stream_url) as ws:
                    self._ws = ws
                    logger.info(f"WebSocket connected: {streams}")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        data = json.loads(message)
                        kline = data.get('k', {})
                        
                        candle = {
                            'symbol': kline.get('s'),
                            'interval': kline.get('i'),
                            'open_time': datetime.fromtimestamp(kline.get('t', 0) / 1000),
                            'open': float(kline.get('o', 0)),
                            'high': float(kline.get('h', 0)),
                            'low': float(kline.get('l', 0)),
                            'close': float(kline.get('c', 0)),
                            'volume': float(kline.get('v', 0)),
                            'is_closed': kline.get('x', False)  # 캔들 마감 여부
                        }
                        
                        await callback(kline.get('i'), candle)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
    
    async def close(self) -> None:
        """연결 종료"""
        self._running = False
        if self._ws:
            await self._ws.close()
    
    # 동기 버전 메서드 (간단한 사용을 위해)
    def get_klines_sync(
        self,
        interval: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """동기 버전 캔들 데이터 조회"""
        return asyncio.run(self.get_klines(interval, limit))
    
    def get_current_price_sync(self) -> float:
        """동기 버전 현재 가격 조회"""
        return asyncio.run(self.get_current_price())


class MockBinanceClient(BinanceClient):
    """테스트용 Mock 클라이언트"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol)
        self._mock_data: Dict[str, pd.DataFrame] = {}
    
    def set_mock_data(self, interval: str, data: pd.DataFrame) -> None:
        """테스트 데이터 설정"""
        self._mock_data[interval] = data
    
    async def get_klines(
        self,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Mock 데이터 반환"""
        if interval in self._mock_data:
            return self._mock_data[interval].tail(limit)
        
        # 없으면 랜덤 데이터 생성
        return self._generate_random_data(limit)
    
    def _generate_random_data(self, limit: int) -> pd.DataFrame:
        """랜덤 테스트 데이터 생성"""
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')
        
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(limit) * 100)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(limit) * 50),
            'low': prices - np.abs(np.random.randn(limit) * 50),
            'close': prices + np.random.randn(limit) * 30,
            'volume': np.random.randint(100, 1000, limit).astype(float)
        }, index=dates)
        
        return df
