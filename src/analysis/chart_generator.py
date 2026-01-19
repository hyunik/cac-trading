"""
차트 이미지 생성 모듈
- 일봉/주봉 캔들차트 생성
- 볼린저밴드, 이동평균선 오버레이
- 시그널 캔들 마커 표시
"""

import os
import io
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 서버용 백엔드
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

logger = logging.getLogger(__name__)


class ChartGenerator:
    """캔들차트 이미지 생성기"""
    
    def __init__(self, output_dir: str = "/tmp/cac_charts"):
        """
        Args:
            output_dir: 차트 이미지 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 스타일 설정
        self.style = {
            'up_color': '#26a69a',      # 상승 캔들 (녹색)
            'down_color': '#ef5350',    # 하락 캔들 (빨강)
            'bb_color': '#2196f3',      # 볼린저밴드 (파랑)
            'ma20_color': '#ff9800',    # MA20 (주황)
            'ma50_color': '#9c27b0',    # MA50 (보라)
            'signal_color': '#ffeb3b',  # 시그널 마커 (노랑)
            'bg_color': '#1e1e1e',      # 배경 (다크)
            'text_color': '#ffffff',    # 텍스트 (흰색)
            'grid_color': '#333333',    # 그리드
        }
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signals: Optional[List[Dict]] = None,
        show_bb: bool = True,
        show_ma: bool = True,
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        캔들차트 이미지 생성
        
        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)
            symbol: 심볼명
            timeframe: 타임프레임 (1d, 1w 등)
            signals: 시그널 마커 리스트 [{'date': datetime, 'type': 'buy'|'sell'}]
            show_bb: 볼린저밴드 표시 여부
            show_ma: 이동평균선 표시 여부
            title: 차트 제목
        
        Returns:
            저장된 이미지 파일 경로 또는 None
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib가 설치되지 않았습니다.")
            return None
        
        if df.empty:
            logger.warning(f"[{symbol}] 빈 데이터프레임")
            return None
        
        # 데이터 준비
        df = df.copy()
        df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
        
        # mplfinance 사용 가능 시
        if HAS_MPLFINANCE:
            return self._generate_with_mplfinance(df, symbol, timeframe, signals, show_bb, show_ma, title)
        else:
            return self._generate_with_matplotlib(df, symbol, timeframe, signals, show_bb, show_ma, title)
    
    def _generate_with_mplfinance(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signals: Optional[List[Dict]],
        show_bb: bool,
        show_ma: bool,
        title: Optional[str]
    ) -> Optional[str]:
        """mplfinance를 사용한 차트 생성"""
        try:
            # 스타일 설정
            mc = mpf.make_marketcolors(
                up=self.style['up_color'],
                down=self.style['down_color'],
                edge='inherit',
                wick='inherit',
                volume='inherit'
            )
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor=self.style['grid_color'],
                facecolor=self.style['bg_color'],
                edgecolor=self.style['text_color'],
                figcolor=self.style['bg_color'],
                rc={'axes.labelcolor': self.style['text_color'],
                    'xtick.color': self.style['text_color'],
                    'ytick.color': self.style['text_color']}
            )
            
            # 추가 플롯 준비
            addplots = []
            
            # 볼린저밴드 계산 및 추가
            if show_bb and len(df) >= 20:
                df['bb_middle'] = df['close'].rolling(20).mean()
                df['bb_std'] = df['close'].rolling(20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
                
                addplots.append(mpf.make_addplot(df['bb_upper'], color=self.style['bb_color'], linestyle='--', width=0.8))
                addplots.append(mpf.make_addplot(df['bb_middle'], color=self.style['bb_color'], width=0.8))
                addplots.append(mpf.make_addplot(df['bb_lower'], color=self.style['bb_color'], linestyle='--', width=0.8))
            
            # 이동평균선 추가
            if show_ma:
                if len(df) >= 20:
                    df['ma20'] = df['close'].rolling(20).mean()
                    addplots.append(mpf.make_addplot(df['ma20'], color=self.style['ma20_color'], width=1))
                if len(df) >= 50:
                    df['ma50'] = df['close'].rolling(50).mean()
                    addplots.append(mpf.make_addplot(df['ma50'], color=self.style['ma50_color'], width=1))
            
            # 시그널 마커 추가
            if signals:
                buy_signals = pd.Series(index=df.index, dtype=float)
                sell_signals = pd.Series(index=df.index, dtype=float)
                
                for sig in signals:
                    sig_date = pd.to_datetime(sig['date'])
                    if sig_date in df.index:
                        if sig['type'] == 'buy':
                            buy_signals[sig_date] = df.loc[sig_date, 'low'] * 0.99
                        else:
                            sell_signals[sig_date] = df.loc[sig_date, 'high'] * 1.01
                
                if buy_signals.notna().any():
                    addplots.append(mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=100, color='lime'))
                if sell_signals.notna().any():
                    addplots.append(mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=100, color='red'))
            
            # 차트 생성
            chart_title = title or f"{symbol} {timeframe.upper()}"
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=s,
                title=chart_title,
                ylabel='Price',
                volume=True,
                addplot=addplots if addplots else None,
                figsize=(12, 8),
                returnfig=True
            )
            
            # 범례 추가
            if show_bb or show_ma:
                legend_items = []
                if show_bb:
                    legend_items.append('BB(20,2)')
                if show_ma and len(df) >= 20:
                    legend_items.append('MA20')
                if show_ma and len(df) >= 50:
                    legend_items.append('MA50')
                if legend_items:
                    axes[0].legend(legend_items, loc='upper left', facecolor=self.style['bg_color'], 
                                   labelcolor=self.style['text_color'])
            
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=self.style['bg_color'])
            plt.close(fig)
            
            logger.info(f"[{symbol}] 차트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"[{symbol}] mplfinance 차트 생성 오류: {e}")
            return None
    
    def _generate_with_matplotlib(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signals: Optional[List[Dict]],
        show_bb: bool,
        show_ma: bool,
        title: Optional[str]
    ) -> Optional[str]:
        """순수 matplotlib을 사용한 차트 생성 (mplfinance 없을 때)"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                            gridspec_kw={'height_ratios': [3, 1]},
                                            facecolor=self.style['bg_color'])
            ax1.set_facecolor(self.style['bg_color'])
            ax2.set_facecolor(self.style['bg_color'])
            
            # 캔들차트 그리기
            width = 0.6
            for i, (idx, row) in enumerate(df.iterrows()):
                color = self.style['up_color'] if row['close'] >= row['open'] else self.style['down_color']
                
                # 몸통
                ax1.add_patch(Rectangle(
                    (i - width/2, min(row['open'], row['close'])),
                    width,
                    abs(row['close'] - row['open']) or 0.01,
                    facecolor=color,
                    edgecolor=color
                ))
                
                # 꼬리
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 볼린저밴드
            if show_bb and len(df) >= 20:
                bb_mid = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                bb_upper = bb_mid + 2 * bb_std
                bb_lower = bb_mid - 2 * bb_std
                
                x = range(len(df))
                ax1.plot(x, bb_upper, color=self.style['bb_color'], linestyle='--', linewidth=0.8, label='BB Upper')
                ax1.plot(x, bb_mid, color=self.style['bb_color'], linewidth=0.8, label='BB Mid')
                ax1.plot(x, bb_lower, color=self.style['bb_color'], linestyle='--', linewidth=0.8, label='BB Lower')
                ax1.fill_between(x, bb_lower, bb_upper, alpha=0.1, color=self.style['bb_color'])
            
            # 이동평균선
            if show_ma:
                x = range(len(df))
                if len(df) >= 20:
                    ma20 = df['close'].rolling(20).mean()
                    ax1.plot(x, ma20, color=self.style['ma20_color'], linewidth=1, label='MA20')
                if len(df) >= 50:
                    ma50 = df['close'].rolling(50).mean()
                    ax1.plot(x, ma50, color=self.style['ma50_color'], linewidth=1, label='MA50')
            
            # 거래량
            colors = [self.style['up_color'] if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                      else self.style['down_color'] for i in range(len(df))]
            ax2.bar(range(len(df)), df['volume'], color=colors, width=0.8)
            
            # 스타일 설정
            chart_title = title or f"{symbol} {timeframe.upper()}"
            ax1.set_title(chart_title, color=self.style['text_color'], fontsize=14)
            ax1.set_ylabel('Price', color=self.style['text_color'])
            ax2.set_ylabel('Volume', color=self.style['text_color'])
            
            ax1.tick_params(colors=self.style['text_color'])
            ax2.tick_params(colors=self.style['text_color'])
            ax1.grid(True, color=self.style['grid_color'], alpha=0.5)
            ax2.grid(True, color=self.style['grid_color'], alpha=0.5)
            
            if show_bb or show_ma:
                ax1.legend(loc='upper left', facecolor=self.style['bg_color'], 
                          labelcolor=self.style['text_color'])
            
            # X축 레이블 설정
            tick_positions = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
            tick_labels = [df.index[i].strftime('%m/%d') for i in tick_positions if i < len(df)]
            ax2.set_xticks([p for p in tick_positions if p < len(df)])
            ax2.set_xticklabels(tick_labels)
            
            ax1.set_xlim(-1, len(df))
            ax2.set_xlim(-1, len(df))
            
            plt.tight_layout()
            
            # 저장
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=self.style['bg_color'])
            plt.close(fig)
            
            logger.info(f"[{symbol}] 차트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"[{symbol}] matplotlib 차트 생성 오류: {e}")
            return None
    
    def generate_multi_timeframe_chart(
        self,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        symbol: str,
        signals_daily: Optional[List[Dict]] = None,
        signals_weekly: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        일봉/주봉 통합 차트 생성
        
        Returns:
            저장된 이미지 파일 경로
        """
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=self.style['bg_color'])
            
            # 일봉 차트
            self._draw_simple_candles(ax1, df_daily, f"{symbol} Daily")
            
            # 주봉 차트
            self._draw_simple_candles(ax2, df_weekly, f"{symbol} Weekly")
            
            plt.tight_layout()
            
            filename = f"{symbol}_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=self.style['bg_color'])
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            logger.error(f"[{symbol}] 통합 차트 생성 오류: {e}")
            return None
    
    def _draw_simple_candles(self, ax, df: pd.DataFrame, title: str):
        """간단한 캔들 차트 그리기"""
        ax.set_facecolor(self.style['bg_color'])
        
        for i, (idx, row) in enumerate(df.iterrows()):
            color = self.style['up_color'] if row['close'] >= row['open'] else self.style['down_color']
            ax.add_patch(Rectangle(
                (i - 0.3, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']) or 0.01,
                facecolor=color,
                edgecolor=color
            ))
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        
        # 볼린저밴드
        if len(df) >= 20:
            bb_mid = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            x = range(len(df))
            ax.plot(x, bb_mid + 2*bb_std, color=self.style['bb_color'], linestyle='--', linewidth=0.8)
            ax.plot(x, bb_mid, color=self.style['bb_color'], linewidth=0.8)
            ax.plot(x, bb_mid - 2*bb_std, color=self.style['bb_color'], linestyle='--', linewidth=0.8)
        
        ax.set_title(title, color=self.style['text_color'], fontsize=12)
        ax.tick_params(colors=self.style['text_color'])
        ax.grid(True, color=self.style['grid_color'], alpha=0.5)
        ax.set_xlim(-1, len(df))
    
    def cleanup_old_charts(self, max_age_hours: int = 24):
        """오래된 차트 파일 정리"""
        import time
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        
        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                logger.debug(f"삭제된 차트: {filename}")
