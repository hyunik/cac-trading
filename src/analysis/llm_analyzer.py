"""
LLM 분석 모듈
- 일봉/주봉 기반 CAC 분석
- 각 코인별 매수/매도/관망 의견 제시
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CACAnalysisResult:
    """CAC 분석 결과"""
    symbol: str
    timeframe: str
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 ~ 1.0
    analysis: str  # LLM 분석 텍스트
    key_points: List[str]  # 주요 포인트
    timestamp: datetime


class LLMAnalyzer:
    """LLM 기반 CAC 분석기"""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """
        Args:
            api_key: LLM API 키
            provider: 'openai' 또는 'gemini'
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.provider = provider
        
        if not self.api_key:
            logger.warning("LLM API 키가 설정되지 않았습니다. 기본 분석만 제공됩니다.")
    
    async def analyze_coin(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        signals_daily: Optional[List[Dict]] = None,
        signals_weekly: Optional[List[Dict]] = None,
        bb_data_daily: Optional[Dict] = None,
        bb_data_weekly: Optional[Dict] = None
    ) -> CACAnalysisResult:
        """
        코인별 CAC 분석 수행
        
        Args:
            symbol: 코인 심볼
            df_daily: 일봉 데이터
            df_weekly: 주봉 데이터
            signals_daily: 일봉 시그널
            signals_weekly: 주봉 시그널
            bb_data_daily: 일봉 볼린저밴드 데이터
            bb_data_weekly: 주봉 볼린저밴드 데이터
        
        Returns:
            CACAnalysisResult
        """
        # 기술적 분석 데이터 준비
        analysis_data = self._prepare_analysis_data(
            symbol, df_daily, df_weekly, 
            signals_daily, signals_weekly,
            bb_data_daily, bb_data_weekly
        )
        
        # LLM API 호출 또는 기본 분석
        if self.api_key:
            analysis_text = await self._call_llm_api(analysis_data)
        else:
            analysis_text = self._generate_basic_analysis(analysis_data)
        
        # 결과 파싱
        result = self._parse_analysis(symbol, analysis_data, analysis_text)
        return result
    
    def _prepare_analysis_data(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        signals_daily: Optional[List[Dict]],
        signals_weekly: Optional[List[Dict]],
        bb_data_daily: Optional[Dict],
        bb_data_weekly: Optional[Dict]
    ) -> Dict[str, Any]:
        """분석 데이터 준비"""
        data = {'symbol': symbol}
        
        # 일봉 분석
        if not df_daily.empty:
            recent_daily = df_daily.tail(20)
            data['daily'] = {
                'current_price': float(df_daily['close'].iloc[-1]),
                'prev_close': float(df_daily['close'].iloc[-2]) if len(df_daily) > 1 else None,
                'change_1d': float((df_daily['close'].iloc[-1] / df_daily['close'].iloc[-2] - 1) * 100) if len(df_daily) > 1 else 0,
                'change_7d': float((df_daily['close'].iloc[-1] / df_daily['close'].iloc[-7] - 1) * 100) if len(df_daily) >= 7 else 0,
                'high_7d': float(df_daily['high'].tail(7).max()),
                'low_7d': float(df_daily['low'].tail(7).min()),
                'volume_avg': float(df_daily['volume'].tail(20).mean()),
                'volume_last': float(df_daily['volume'].iloc[-1]),
                'sma20': float(df_daily['close'].rolling(20).mean().iloc[-1]) if len(df_daily) >= 20 else None,
                'sma50': float(df_daily['close'].rolling(50).mean().iloc[-1]) if len(df_daily) >= 50 else None
            }
            data['signals_daily'] = len(signals_daily) if signals_daily else 0
        
        # 주봉 분석
        if not df_weekly.empty:
            data['weekly'] = {
                'current_price': float(df_weekly['close'].iloc[-1]),
                'change_1w': float((df_weekly['close'].iloc[-1] / df_weekly['close'].iloc[-2] - 1) * 100) if len(df_weekly) > 1 else 0,
                'change_4w': float((df_weekly['close'].iloc[-1] / df_weekly['close'].iloc[-4] - 1) * 100) if len(df_weekly) >= 4 else 0,
                'high_4w': float(df_weekly['high'].tail(4).max()),
                'low_4w': float(df_weekly['low'].tail(4).min())
            }
            data['signals_weekly'] = len(signals_weekly) if signals_weekly else 0
        
        # 볼린저밴드 상태
        if bb_data_daily:
            data['bb_daily'] = bb_data_daily
        if bb_data_weekly:
            data['bb_weekly'] = bb_data_weekly
        
        # 추세 판단
        if data.get('daily', {}).get('sma20') and data.get('daily', {}).get('sma50'):
            price = data['daily']['current_price']
            sma20 = data['daily']['sma20']
            sma50 = data['daily']['sma50']
            
            if price > sma20 > sma50:
                data['trend'] = 'BULLISH'
            elif price < sma20 < sma50:
                data['trend'] = 'BEARISH'
            else:
                data['trend'] = 'NEUTRAL'
        else:
            data['trend'] = 'NEUTRAL'
        
        return data
    
    async def _call_llm_api(self, analysis_data: Dict[str, Any]) -> str:
        """LLM API 호출"""
        prompt = self._build_prompt(analysis_data)
        
        try:
            if self.provider == "openai":
                return await self._call_openai(prompt)
            elif self.provider == "gemini":
                return await self._call_gemini(prompt)
            else:
                return self._generate_basic_analysis(analysis_data)
        except Exception as e:
            logger.error(f"LLM API 호출 오류: {e}")
            return self._generate_basic_analysis(analysis_data)
    
    async def _call_openai(self, prompt: str) -> str:
        """OpenAI API 호출"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "당신은 전문 암호화폐 기술 분석가입니다. 간결하고 명확한 분석을 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    logger.error(f"OpenAI API 오류: {response.status} - {error}")
                    return ""
    
    async def _call_gemini(self, prompt: str) -> str:
        """Gemini API 호출"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.7
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    error = await response.text()
                    logger.error(f"Gemini API 오류: {response.status} - {error}")
                    return ""
    
    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """LLM 프롬프트 생성"""
        symbol = data['symbol']
        daily = data.get('daily', {})
        weekly = data.get('weekly', {})
        trend = data.get('trend', 'NEUTRAL')
        
        prompt = f"""
{symbol} 암호화폐에 대한 기술적 분석을 제공해주세요.

## 데이터
- 현재가: ${daily.get('current_price', 0):,.2f}
- 일간 변동: {daily.get('change_1d', 0):+.2f}%
- 주간 변동: {daily.get('change_7d', 0):+.2f}%
- 4주 변동: {weekly.get('change_4w', 0):+.2f}%
- SMA20: ${daily.get('sma20', 0):,.2f}
- SMA50: ${daily.get('sma50', 0):,.2f}
- 추세: {trend}
- 일봉 시그널: {data.get('signals_daily', 0)}개
- 주봉 시그널: {data.get('signals_weekly', 0)}개

## 요청
1. 현재 추세 분석 (1줄)
2. 주요 지지/저항 구간
3. 매수/매도/관망 중 하나 추천 (이유 포함)

간결하게 한국어로 200자 이내로 답변해주세요.
"""
        return prompt
    
    def _generate_basic_analysis(self, data: Dict[str, Any]) -> str:
        """기본 분석 생성 (LLM 없이)"""
        symbol = data['symbol']
        daily = data.get('daily', {})
        trend = data.get('trend', 'NEUTRAL')
        
        price = daily.get('current_price', 0)
        change_1d = daily.get('change_1d', 0)
        change_7d = daily.get('change_7d', 0)
        sma20 = daily.get('sma20')
        sma50 = daily.get('sma50')
        
        # 추세 분석
        if trend == 'BULLISH':
            trend_text = "상승 추세"
            signal = "BUY"
        elif trend == 'BEARISH':
            trend_text = "하락 추세"
            signal = "SELL"
        else:
            trend_text = "횡보 추세"
            signal = "HOLD"
        
        # 분석 텍스트 생성
        analysis = f"📊 {symbol}: {trend_text}\n"
        analysis += f"• 현재가: ${price:,.2f} ({change_1d:+.2f}%)\n"
        analysis += f"• 7일 변동: {change_7d:+.2f}%\n"
        
        if sma20 and sma50:
            if price > sma20:
                analysis += f"• MA20(${sma20:,.2f}) 위에서 거래 중 ✅\n"
            else:
                analysis += f"• MA20(${sma20:,.2f}) 아래에서 거래 중 ⚠️\n"
        
        analysis += f"• 추천: {signal}"
        
        return analysis
    
    def _parse_analysis(
        self, 
        symbol: str, 
        data: Dict[str, Any], 
        analysis_text: str
    ) -> CACAnalysisResult:
        """분석 결과 파싱"""
        trend = data.get('trend', 'NEUTRAL')
        
        # 시그널 판단
        if 'BUY' in analysis_text.upper() or '매수' in analysis_text:
            signal = 'BUY'
            confidence = 0.7
        elif 'SELL' in analysis_text.upper() or '매도' in analysis_text:
            signal = 'SELL'
            confidence = 0.7
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # 주요 포인트 추출
        key_points = []
        if data.get('daily', {}).get('change_7d', 0) > 10:
            key_points.append("📈 7일간 강한 상승")
        elif data.get('daily', {}).get('change_7d', 0) < -10:
            key_points.append("📉 7일간 강한 하락")
        
        if data.get('signals_daily', 0) > 0:
            key_points.append(f"🎯 일봉 시그널 {data['signals_daily']}개 감지")
        
        return CACAnalysisResult(
            symbol=symbol,
            timeframe='1D',
            trend=trend,
            signal=signal,
            confidence=confidence,
            analysis=analysis_text,
            key_points=key_points,
            timestamp=datetime.now()
        )
    
    async def analyze_all_coins(
        self,
        coins_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> List[CACAnalysisResult]:
        """
        모든 코인 분석 수행
        
        Args:
            coins_data: {symbol: {'daily': df, 'weekly': df, 'signals': [...]}}
        
        Returns:
            분석 결과 리스트
        """
        tasks = []
        for symbol, data in coins_data.items():
            task = self.analyze_coin(
                symbol=symbol,
                df_daily=data.get('daily', pd.DataFrame()),
                df_weekly=data.get('weekly', pd.DataFrame()),
                signals_daily=data.get('signals_daily'),
                signals_weekly=data.get('signals_weekly'),
                bb_data_daily=data.get('bb_daily'),
                bb_data_weekly=data.get('bb_weekly')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return list(results)
