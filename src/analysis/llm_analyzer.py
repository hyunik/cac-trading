"""
LLM 분석 모듈 - 캔들·거래량 기반 크립토 차트 분석기
- 주봉 → 일봉 순서로 분석
- 캔들 형태와 거래량 변화를 핵심 근거로 분석
- BTC와 유사한 알트코인 식별
- 롱/숏 추세 국면 판단
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CACAnalysisResult:
    """CAC 분석 결과"""
    symbol: str
    timeframe: str
    trend: str  # 'LONG_BIAS', 'SHORT_BIAS', 'NEUTRAL'
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 ~ 1.0
    analysis: str  # LLM 분석 텍스트
    key_points: List[str]  # 주요 포인트
    timestamp: datetime


@dataclass  
class MarketOverview:
    """전체 시장 분석 결과"""
    btc_trend: str
    market_phase: str
    btc_similar_coins: List[str]
    analysis: str


class LLMAnalyzer:
    """캔들·거래량 기반 LLM 분석기"""
    
    # 시스템 프롬프트 - 캔들·거래량 분석 전문가
    SYSTEM_PROMPT = """당신은 "캔들·거래량 기반 크립토 차트 분석기"다.

[최상위 원칙]
1) 모든 분석의 1차 근거는 "캔들 + 거래량"이다.
2) 구조·이론·개념을 끼워 맞추지 않는다. 오직 차트에 '보이는 것'만으로 설명한다.
3) 가장 최근 캔들일수록 분석 비중이 높다.
4) 확정적 표현, 수익 보장, 단정적 진입 유도는 금지한다. 모든 결론은 조건부 시나리오로 제시한다.

[분석 우선순위]
1순위: 가장 최근 캔들의 형태와 크기
2순위: 해당 캔들의 거래량 변화 (직전 대비)
3순위: 최근 1~3개 캔들의 연속성
4순위: 그 위의 추세 환경(주봉 → 일봉)

[시간 가중치 규칙]
- 최근 1개 캔들: 결정적 시그널 후보
- 최근 2~3개 캔들: 힘의 연속성 또는 전환 판단
- 최근 4~10개 캔들: 추세 환경 설명

[캔들 해석 규칙]
- 큰 몸통 캔들: 해당 방향으로의 명확한 힘
- 긴 꼬리: 해당 가격대에서의 거절 또는 흡수
- 장악형(Engulfing): 이전 흐름을 압도한 힘
- 점점 줄어드는 캔들 크기: 추세 에너지 약화 가능성

[거래량 해석 규칙]
- 큰 양봉 + 거래량 증가 → 상승 의지에 실제 참여
- 큰 음봉 + 거래량 증가 → 매도 우위 강화
- 긴 꼬리 + 대량 거래량 → 강한 거절 또는 흡수 가능성
- 캔들은 큰데 거래량이 작음 → 신뢰도 낮음, 추격 경계"""

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
    
    async def analyze_market_overview(
        self,
        all_coins_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> MarketOverview:
        """
        전체 시장 분석 - BTC 기준 + 알트코인 상관관계
        
        Args:
            all_coins_data: {symbol: {'daily': df, 'weekly': df}}
        
        Returns:
            MarketOverview
        """
        btc_data = all_coins_data.get('BTCUSDT', {})
        
        # BTC 캔들 데이터 분석
        btc_analysis = self._analyze_candle_volume(
            btc_data.get('daily', pd.DataFrame()),
            btc_data.get('weekly', pd.DataFrame())
        )
        
        # 각 알트코인과 BTC 유사도 계산
        similarities = {}
        for symbol, coin_data in all_coins_data.items():
            if symbol == 'BTCUSDT':
                continue
            df_daily = coin_data.get('daily', pd.DataFrame())
            if not df_daily.empty and not btc_data.get('daily', pd.DataFrame()).empty:
                corr = self._calculate_correlation(
                    btc_data['daily'], df_daily
                )
                similarities[symbol] = corr
        
        # BTC와 가장 유사한 코인 3개
        btc_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        btc_similar_coins = [s[0] for s in btc_similar]
        
        # 시장 국면 판단
        market_phase = self._determine_market_phase(btc_analysis)
        
        # LLM으로 종합 분석 생성
        if self.api_key:
            analysis = await self._generate_market_overview(
                btc_analysis, btc_similar_coins, similarities, all_coins_data
            )
        else:
            analysis = self._basic_market_overview(btc_analysis, btc_similar_coins)
        
        return MarketOverview(
            btc_trend=btc_analysis.get('trend', 'NEUTRAL'),
            market_phase=market_phase,
            btc_similar_coins=btc_similar_coins,
            analysis=analysis
        )
    
    def _analyze_candle_volume(
        self, 
        df_daily: pd.DataFrame, 
        df_weekly: pd.DataFrame
    ) -> Dict[str, Any]:
        """캔들 + 거래량 분석"""
        result = {
            'trend': 'NEUTRAL',
            'daily_candles': [],
            'weekly_candles': [],
            'volume_trend': 'NORMAL'
        }
        
        if df_daily.empty:
            return result
        
        # 최근 일봉 분석 (10개)
        recent_daily = df_daily.tail(10)
        for i, (idx, row) in enumerate(recent_daily.iterrows()):
            candle = self._parse_candle(row, df_daily, i)
            result['daily_candles'].append(candle)
        
        # 최근 주봉 분석 (4개)
        if not df_weekly.empty:
            recent_weekly = df_weekly.tail(4)
            for i, (idx, row) in enumerate(recent_weekly.iterrows()):
                candle = self._parse_candle(row, df_weekly, i)
                result['weekly_candles'].append(candle)
        
        # 트렌드 판단 (최근 캔들 기준)
        if result['daily_candles']:
            last_candle = result['daily_candles'][-1]
            prev_candles = result['daily_candles'][-3:-1]
            
            # 최근 캔들 방향과 거래량으로 판단
            if last_candle['type'] == 'BULLISH' and last_candle['volume_ratio'] > 1.2:
                result['trend'] = 'LONG_BIAS'
            elif last_candle['type'] == 'BEARISH' and last_candle['volume_ratio'] > 1.2:
                result['trend'] = 'SHORT_BIAS'
            else:
                # 최근 3개 캔들 종합
                bullish_count = sum(1 for c in prev_candles if c['type'] == 'BULLISH')
                if bullish_count >= 2:
                    result['trend'] = 'LONG_BIAS'
                elif bullish_count == 0:
                    result['trend'] = 'SHORT_BIAS'
                else:
                    result['trend'] = 'NEUTRAL'
        
        # 거래량 트렌드
        if len(df_daily) >= 20:
            vol_avg_20 = df_daily['volume'].tail(20).mean()
            vol_recent = df_daily['volume'].tail(5).mean()
            vol_ratio = vol_recent / vol_avg_20 if vol_avg_20 > 0 else 1
            
            if vol_ratio > 1.5:
                result['volume_trend'] = 'INCREASING'
            elif vol_ratio < 0.7:
                result['volume_trend'] = 'DECREASING'
            else:
                result['volume_trend'] = 'NORMAL'
        
        return result
    
    def _parse_candle(self, row: pd.Series, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """캔들 데이터 파싱"""
        open_price = float(row['open'])
        close_price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        volume = float(row['volume'])
        
        # 캔들 타입
        body = abs(close_price - open_price)
        upper_wick = high - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low
        total_range = high - low if high > low else 0.0001
        
        candle_type = 'BULLISH' if close_price >= open_price else 'BEARISH'
        
        # 캔들 패턴 감지
        pattern = 'NORMAL'
        if body < total_range * 0.1:
            pattern = 'DOJI'
        elif lower_wick > body * 2 and upper_wick < body * 0.5:
            pattern = 'HAMMER'
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            pattern = 'SHOOTING_STAR'
        elif body > total_range * 0.7:
            pattern = 'STRONG_BODY'
        
        # 거래량 비교 (20일 평균 대비)
        vol_avg = df['volume'].tail(20).mean() if len(df) >= 20 else volume
        volume_ratio = volume / vol_avg if vol_avg > 0 else 1.0
        
        return {
            'type': candle_type,
            'pattern': pattern,
            'body_ratio': body / total_range if total_range > 0 else 0,
            'upper_wick_ratio': upper_wick / total_range if total_range > 0 else 0,
            'lower_wick_ratio': lower_wick / total_range if total_range > 0 else 0,
            'volume_ratio': volume_ratio,
            'change_pct': (close_price / open_price - 1) * 100
        }
    
    def _calculate_correlation(self, df_btc: pd.DataFrame, df_alt: pd.DataFrame) -> float:
        """BTC와 알트코인 상관관계 계산"""
        try:
            if len(df_btc) < 20 or len(df_alt) < 20:
                return 0.0
            
            # 최근 20일 수익률 상관관계
            btc_returns = df_btc['close'].pct_change().tail(20).dropna()
            alt_returns = df_alt['close'].pct_change().tail(20).dropna()
            
            if len(btc_returns) < 10 or len(alt_returns) < 10:
                return 0.0
            
            # 길이 맞추기
            min_len = min(len(btc_returns), len(alt_returns))
            btc_returns = btc_returns.iloc[-min_len:]
            alt_returns = alt_returns.iloc[-min_len:]
            
            correlation = np.corrcoef(btc_returns.values, alt_returns.values)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            logger.error(f"상관관계 계산 오류: {e}")
            return 0.0
    
    def _determine_market_phase(self, btc_analysis: Dict[str, Any]) -> str:
        """BTC 기준 시장 국면 판단"""
        trend = btc_analysis.get('trend', 'NEUTRAL')
        volume_trend = btc_analysis.get('volume_trend', 'NORMAL')
        
        if trend == 'LONG_BIAS':
            if volume_trend == 'INCREASING':
                return "강한 롱 추세 (거래량 증가)"
            else:
                return "롱 추세 (거래량 주의)"
        elif trend == 'SHORT_BIAS':
            if volume_trend == 'INCREASING':
                return "강한 숏 추세 (거래량 증가)"
            else:
                return "숏 추세 (거래량 주의)"
        else:
            return "횡보/관망 국면"
    
    async def _generate_market_overview(
        self,
        btc_analysis: Dict[str, Any],
        btc_similar_coins: List[str],
        similarities: Dict[str, float],
        all_coins_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> str:
        """LLM으로 시장 종합 분석 생성"""
        
        # 최근 캔들 데이터 설명
        daily_desc = self._describe_candles(btc_analysis.get('daily_candles', []), '일봉')
        weekly_desc = self._describe_candles(btc_analysis.get('weekly_candles', []), '주봉')
        
        prompt = f"""
다음은 암호화폐 시장의 캔들·거래량 데이터입니다. 분석해주세요.

## BTC 주봉 분석
{weekly_desc}

## BTC 일봉 분석 (최근 우선)
{daily_desc}

## BTC와 가장 유사한 알트코인
1. {btc_similar_coins[0] if len(btc_similar_coins) > 0 else 'N/A'} (상관관계: {similarities.get(btc_similar_coins[0], 0):.2f})
2. {btc_similar_coins[1] if len(btc_similar_coins) > 1 else 'N/A'} (상관관계: {similarities.get(btc_similar_coins[1], 0) if len(btc_similar_coins) > 1 else 0:.2f})
3. {btc_similar_coins[2] if len(btc_similar_coins) > 2 else 'N/A'} (상관관계: {similarities.get(btc_similar_coins[2], 0) if len(btc_similar_coins) > 2 else 0:.2f})

[출력 포맷 - 고정]

A. 한 줄 요약
→ "주봉 기준 ○○ 환경 속에서, 최근 일봉은 ○○한 힘이 관찰됨"

B. 주봉 분석 (최근 캔들 우선)
- 가장 최근 주봉 캔들의 형태와 거래량
- 중기 관점(2~4주)의 유리/불리 환경

C. 일봉 분석 (가장 최근 캔들 중심)
- 최근 1~3개 일봉의 힘과 거래량
- 추격 / 눌림 / 관망 중 어떤 상태인지

D. 추세 국면 판단
- 롱 추세 / 숏 추세 / 횡보 중 하나
- 판단 근거 (캔들 + 거래량 기반)

E. BTC 유사 알트코인 분석
- BTC와 흡사한 형태의 코인과 그 이유

F. 시나리오 (조건부)
- 상승 시나리오: 어떤 캔들 + 거래량이 나오면
- 하락 시나리오: 어떤 캔들 + 거래량이 나오면

한국어로 400자 이내로 답변해주세요.
"""
        
        try:
            if self.provider == "openai":
                return await self._call_openai(prompt)
            elif self.provider == "gemini":
                return await self._call_gemini(prompt)
            else:
                return self._basic_market_overview(btc_analysis, btc_similar_coins)
        except Exception as e:
            logger.error(f"LLM API 호출 오류: {e}")
            return self._basic_market_overview(btc_analysis, btc_similar_coins)
    
    def _describe_candles(self, candles: List[Dict], timeframe: str) -> str:
        """캔들 데이터를 텍스트로 설명"""
        if not candles:
            return "데이터 없음"
        
        lines = []
        for i, c in enumerate(reversed(candles[-5:])):  # 최근 5개 (역순: 최근이 먼저)
            vol_desc = "대량" if c['volume_ratio'] > 1.5 else "평균" if c['volume_ratio'] > 0.8 else "저량"
            pattern_desc = {
                'DOJI': '도지',
                'HAMMER': '망치형',
                'SHOOTING_STAR': '역망치형',
                'STRONG_BODY': '장대',
                'NORMAL': '일반'
            }.get(c['pattern'], c['pattern'])
            
            if i == 0:
                prefix = "★가장 최근"
            else:
                prefix = f"{i+1}개 전"
            
            lines.append(f"- {prefix}: {c['type']}({pattern_desc}), 변동 {c['change_pct']:+.2f}%, 거래량 {vol_desc}({c['volume_ratio']:.1f}x)")
        
        return "\n".join(lines)
    
    def _basic_market_overview(self, btc_analysis: Dict, btc_similar_coins: List[str]) -> str:
        """기본 시장 분석 (LLM 없이)"""
        trend = btc_analysis.get('trend', 'NEUTRAL')
        volume_trend = btc_analysis.get('volume_trend', 'NORMAL')
        
        if trend == 'LONG_BIAS':
            trend_text = "📈 롱 추세 국면"
            signal = "롱 우위"
        elif trend == 'SHORT_BIAS':
            trend_text = "📉 숏 추세 국면"
            signal = "숏 우위"
        else:
            trend_text = "➡️ 횡보 국면"
            signal = "관망"
        
        vol_text = {
            'INCREASING': '거래량 증가 중',
            'DECREASING': '거래량 감소 중',
            'NORMAL': '거래량 평균'
        }.get(volume_trend, '거래량 평균')
        
        analysis = f"**{trend_text}**\n\n"
        analysis += f"• 시장 상태: {signal}\n"
        analysis += f"• 거래량: {vol_text}\n"
        if btc_similar_coins:
            analysis += f"• BTC 유사 코인: {', '.join(btc_similar_coins[:3])}\n"
        
        return analysis
    
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
        """개별 코인 CAC 분석"""
        
        # 캔들 + 거래량 분석
        analysis_data = self._analyze_candle_volume(df_daily, df_weekly)
        
        # 추가 데이터 준비
        analysis_data['symbol'] = symbol
        analysis_data['current_price'] = float(df_daily['close'].iloc[-1]) if not df_daily.empty else 0
        analysis_data['change_1d'] = float((df_daily['close'].iloc[-1] / df_daily['close'].iloc[-2] - 1) * 100) if len(df_daily) > 1 else 0
        
        # LLM 분석 또는 기본 분석
        if self.api_key:
            analysis_text = await self._generate_coin_analysis(analysis_data)
        else:
            analysis_text = self._generate_basic_coin_analysis(analysis_data)
        
        # 결과 파싱
        trend = analysis_data.get('trend', 'NEUTRAL')
        if trend == 'LONG_BIAS':
            signal = 'BUY'
        elif trend == 'SHORT_BIAS':
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return CACAnalysisResult(
            symbol=symbol,
            timeframe='1D',
            trend=trend,
            signal=signal,
            confidence=0.7 if trend != 'NEUTRAL' else 0.5,
            analysis=analysis_text,
            key_points=[f"추세: {trend}", f"거래량: {analysis_data.get('volume_trend', 'N/A')}"],
            timestamp=datetime.now()
        )
    
    async def _generate_coin_analysis(self, data: Dict[str, Any]) -> str:
        """개별 코인 LLM 분석"""
        symbol = data.get('symbol', 'UNKNOWN')
        daily_desc = self._describe_candles(data.get('daily_candles', []), '일봉')
        weekly_desc = self._describe_candles(data.get('weekly_candles', []), '주봉')
        
        prompt = f"""
{symbol} 캔들·거래량 분석:

## 주봉
{weekly_desc}

## 일봉 (최근 우선)
{daily_desc}

## 현재가: ${data.get('current_price', 0):,.2f} ({data.get('change_1d', 0):+.2f}%)

[출력 포맷]
A. 한 줄 요약 (주봉 환경 + 일봉 상태)
B. 캔들 분석 (최근 캔들 형태와 의미)
C. 거래량 분석 (힘의 크기)
D. 추세 국면: 롱/숏/횡보
E. 조건부 시나리오

150자 이내, 한국어로.
"""
        
        try:
            if self.provider == "openai":
                return await self._call_openai(prompt)
            else:
                return await self._call_gemini(prompt)
        except Exception as e:
            return self._generate_basic_coin_analysis(data)
    
    def _generate_basic_coin_analysis(self, data: Dict[str, Any]) -> str:
        """기본 코인 분석 (LLM 없이)"""
        symbol = data.get('symbol', 'UNKNOWN')
        trend = data.get('trend', 'NEUTRAL')
        volume_trend = data.get('volume_trend', 'NORMAL')
        
        trend_map = {
            'LONG_BIAS': '롱 추세',
            'SHORT_BIAS': '숏 추세', 
            'NEUTRAL': '횡보'
        }
        
        vol_map = {
            'INCREASING': '거래량 ↑',
            'DECREASING': '거래량 ↓',
            'NORMAL': '거래량 보통'
        }
        
        return f"📊 {symbol}: {trend_map.get(trend, '횡보')}\n• {vol_map.get(volume_trend, '거래량 보통')}"
    
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
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.5
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
        
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 800,
                "temperature": 0.5
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
    
    async def analyze_all_coins(
        self,
        coins_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> List[CACAnalysisResult]:
        """모든 코인 분석"""
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
