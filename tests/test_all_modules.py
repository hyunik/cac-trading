"""
트레이딩 시스템 전체 모듈 테스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 테스트 데이터 생성
# ==========================================

def create_test_data(days: int = 30, interval_minutes: int = 60) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성"""
    periods = days * 24 * 60 // interval_minutes
    dates = pd.date_range(end=datetime.now(), periods=periods, freq=f'{interval_minutes}min')
    
    np.random.seed(42)
    base_price = 50000
    returns = np.random.randn(periods) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(periods) * 50),
        'low': prices - np.abs(np.random.randn(periods) * 50),
        'close': prices * (1 + np.random.randn(periods) * 0.001),
        'volume': np.random.randint(100, 5000, periods).astype(float)
    }, index=dates)
    
    # 일부 패턴 추가 (장악형 캔들)
    for i in range(5, len(df), 50):
        # 상승 장악형 패턴 추가
        df.iloc[i-1, df.columns.get_loc('close')] = df.iloc[i-1]['open'] - 100
        df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1]['close'] - 50
        df.iloc[i, df.columns.get_loc('close')] = df.iloc[i-1]['open'] + 150
        df.iloc[i, df.columns.get_loc('volume')] *= 3
    
    return df


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ==========================================
# 1. 시그널 감지 모듈 테스트
# ==========================================

def test_signal_detector():
    print_section("1. 시그널 감지 모듈 테스트")
    
    from src.analysis.signal_detector import SignalDetector, SignalType
    
    detector = SignalDetector(volume_threshold=1.0)
    df = create_test_data(days=10, interval_minutes=60)
    
    signals = detector.detect_all_signals(df)
    print(f"✅ 감지된 시그널: {len(signals)}개")
    
    for signal in signals[:5]:  # 처음 5개만 출력
        print(f"   - {signal.signal_type.value}: {signal.timestamp} @ ${signal.entry_price:,.2f}")
        print(f"     신뢰도: {signal.confidence.value}★, 거래량 비율: {signal.volume_ratio:.1f}x")
    
    # 유효 시그널 필터
    valid = detector.filter_valid_signals(signals)
    print(f"✅ 거래량 검증 통과 시그널: {len(valid)}개")
    
    # 고신뢰도 시그널
    high_conf = detector.get_high_confidence_signals(signals)
    print(f"✅ 고신뢰도 시그널 (★★★★★): {len(high_conf)}개")
    
    return True


# ==========================================
# 2. 볼린저밴드 모듈 테스트
# ==========================================

def test_bollinger_bands():
    print_section("2. 볼린저밴드 모듈 테스트")
    
    from src.analysis.bollinger_bands import BollingerBands, BandPosition
    
    bb = BollingerBands(period=20, std_dev=2.0)
    df = create_test_data(days=5, interval_minutes=60)
    
    # 볼린저밴드 계산
    df_bb = bb.calculate(df)
    print(f"✅ 볼린저밴드 계산 완료")
    print(f"   상단: ${df_bb['bb_upper'].iloc[-1]:,.2f}")
    print(f"   중앙: ${df_bb['bb_middle'].iloc[-1]:,.2f}")
    print(f"   하단: ${df_bb['bb_lower'].iloc[-1]:,.2f}")
    print(f"   밴드폭: {df_bb['bb_bandwidth'].iloc[-1]:.2f}%")
    print(f"   %B: {df_bb['bb_percent_b'].iloc[-1]:.3f}")
    
    # 현재 밴드 정보
    band_data = bb.get_current_band(df)
    if band_data:
        print(f"✅ 현재 위치: {band_data.position.value}")
    
    # 터치 신호
    is_touch, direction, desc = bb.get_band_touch_signal(df)
    print(f"✅ 터치 신호: {is_touch} - {desc}")
    
    return True


# ==========================================
# 3. 피보나치 모듈 테스트
# ==========================================

def test_fibonacci():
    print_section("3. 피보나치 모듈 테스트")
    
    from src.analysis.fibonacci import FibonacciRetracement
    
    fib = FibonacciRetracement()
    
    # 피보나치 레벨 계산
    levels = fib.calculate_levels(swing_high=55000, swing_low=50000, is_uptrend=True)
    print("✅ 피보나치 레벨 (상승 후 되돌림):")
    for level in levels:
        print(f"   {level.label}: ${level.price:,.2f}")
    
    # 복구 구간 테스트
    df = create_test_data(days=3, interval_minutes=60)
    zone = fib.get_recovery_zone(df, stop_loss_triggered_direction="LONG", lookback=20)
    
    if zone:
        print(f"✅ 복구 진입 구간 (롱 손절 후):")
        print(f"   진입 시작 (0.5): ${zone.entry_zone_start:,.2f}")
        print(f"   진입 끝 (0.618): ${zone.entry_zone_end:,.2f}")
        print(f"   손절가: ${zone.stop_loss:,.2f}")
        print(f"   방향: {'롱' if zone.is_bullish else '숏'}")
    
    return True


# ==========================================
# 4. 15분봉 전략 테스트
# ==========================================

def test_strategy_15m():
    print_section("4. 15분봉 전략 테스트")
    
    from src.strategies.strategy_15m import Strategy15M, DailyBias
    
    strategy = Strategy15M()
    
    # 일봉 데이터로 바이어스 설정
    df_daily = create_test_data(days=30, interval_minutes=1440)
    strategy.set_daily_bias(df_daily, datetime.now())
    print(f"✅ 데일리 바이어스: {strategy.state.daily_bias.value}")
    
    # 15분봉 데이터로 진입 신호 체크
    df_15m = create_test_data(days=3, interval_minutes=15)
    current_price = df_15m['close'].iloc[-1]
    
    signal = strategy.check_entry_signal(df_15m, current_price, datetime.now())
    if signal:
        print(f"✅ 진입 신호 발생:")
        print(f"   방향: {signal['direction']}")
        print(f"   진입가: ${signal['entry_price']:,.2f}")
        print(f"   손절가: ${signal['stop_loss']:,.2f}")
        print(f"   비중: {signal['size_percent']}%")
        
        # 진입 실행
        position = strategy.enter_position(signal)
        print(f"✅ 포지션 진입 완료: 피라미딩 레벨 {position.pyramid_level.value}")
    else:
        print("ℹ️ 현재 진입 신호 없음 (바이어스/시그널 불일치 또는 시그널 없음)")
    
    # 전략 요약
    summary = strategy.get_summary()
    print(f"✅ 전략 요약: {summary}")
    
    return True


# ==========================================
# 5. 1시간봉 전략 테스트
# ==========================================

def test_strategy_1h():
    print_section("5. 1시간봉 전략 테스트")
    
    from src.strategies.strategy_1h import Strategy1H
    
    strategy = Strategy1H()
    
    # 1시간봉 데이터
    df_1h = create_test_data(days=7, interval_minutes=60)
    
    # 볼린저밴드 하단 터치 시뮬레이션
    # 마지막 캔들을 하단 근처로 조정
    df_1h.iloc[-1, df_1h.columns.get_loc('close')] = df_1h['close'].iloc[-1] * 0.95
    df_1h.iloc[-1, df_1h.columns.get_loc('low')] = df_1h['low'].iloc[-1] * 0.94
    
    signal = strategy.check_entry_signal(df_1h, datetime.now())
    if signal:
        print(f"✅ 진입 신호 발생:")
        print(f"   방향: {signal['direction']}")
        print(f"   진입가: ${signal['entry_price']:,.2f}")
        print(f"   비중: {signal['size_percent']}%")
        print(f"   밴드 설명: {signal['band_description']}")
        
        # 진입 실행
        position = strategy.enter_position(signal)
        print(f"✅ 포지션 진입 완료")
    else:
        print("ℹ️ 현재 진입 신호 없음 (볼밴 터치 없음 또는 시그널 불일치)")
    
    summary = strategy.get_summary()
    print(f"✅ 전략 요약: {summary}")
    
    return True


# ==========================================
# 6. 포지션 관리자 테스트
# ==========================================

def test_position_manager():
    print_section("6. 포지션 관리자 테스트")
    
    from src.core.position_manager import PositionManager
    
    pm = PositionManager(initial_capital=10000)
    
    # 거래 생성
    trade1 = pm.create_trade(
        strategy="15분봉",
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=50000,
        size_percent=10
    )
    print(f"✅ 거래 생성: {trade1.id}")
    
    # 거래 청산 (익절)
    closed = pm.close_trade(trade1.id, exit_price=51500)  # +3%
    print(f"✅ 거래 청산: PnL {closed.pnl_percent:.2f}%")
    
    # 통계
    stats = pm.get_statistics()
    print(f"✅ 통계: 승률 {stats['win_rate']:.1f}%, 거래 {stats['total_trades']}건")
    
    # 손익
    pnl = pm.get_total_pnl()
    print(f"✅ 현재 자본: ${pnl['current_capital']:,.2f} (ROI: {pnl['roi']:.2f}%)")
    
    return True


# ==========================================
# 7. 리스크 관리자 테스트
# ==========================================

def test_risk_manager():
    print_section("7. 리스크 관리자 테스트")
    
    from src.core.risk_manager import RiskManager, RiskLimits
    
    rm = RiskManager(RiskLimits(
        max_position_size=90,
        max_daily_loss=10,
        max_consecutive_losses=3
    ))
    
    # 진입 허용 체크
    allowed, reason = rm.check_entry_allowed(position_size=25, stop_loss_percent=5)
    print(f"✅ 진입 허용: {allowed} - {reason}")
    
    # 포지션 추가
    rm.update_position_size(25)
    
    # 손실 기록
    rm.record_trade_result(pnl_percent=-3, size_percent=25)
    
    status = rm.get_status()
    print(f"✅ 리스크 상태:")
    print(f"   거래 허용: {status['is_trading_allowed']}")
    print(f"   리스크 레벨: {status['risk_level']}")
    print(f"   일일 손익: {status['daily_pnl']:.2f}%")
    print(f"   연속 손실: {status['consecutive_losses']}회")
    
    return True


# ==========================================
# 8. Discord 알림 테스트 (Mock)
# ==========================================

def test_discord_notifier():
    print_section("8. Discord 알림 모듈 테스트 (Mock)")
    
    from src.notifications.discord_notifier import DiscordNotifier
    
    # Mock URL (실제로 전송하지 않음)
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test/test")
    
    # 거래 로그 추가 (테스트용)
    notifier._trade_log.append({
        'type': 'TAKE_PROFIT',
        'strategy': '15분봉 추세 피라미딩',
        'direction': 'LONG',
        'symbol': 'BTCUSDT',
        'entry_price': 50000,
        'exit_price': 51500,
        'pnl_percent': 3.0,
        'size_percent': 10,
        'timestamp': datetime.now()
    })
    notifier._trade_log.append({
        'type': 'STOP_LOSS',
        'strategy': '1시간봉 볼밴 역추세',
        'direction': 'SHORT',
        'symbol': 'BTCUSDT',
        'entry_price': 52000,
        'exit_price': 52500,
        'pnl_percent': -0.96,
        'size_percent': 25,
        'timestamp': datetime.now()
    })
    
    # 통계 계산 테스트
    stats_15m = notifier._calculate_strategy_stats(notifier._trade_log, '15분봉 추세 피라미딩')
    stats_1h = notifier._calculate_strategy_stats(notifier._trade_log, '1시간봉 볼밴 역추세')
    
    print(f"✅ 15분봉 전략 통계: 거래 {stats_15m['total_trades']}건, 수익률 {stats_15m['total_pnl']:.2f}%")
    print(f"✅ 1시간봉 전략 통계: 거래 {stats_1h['total_trades']}건, 수익률 {stats_1h['total_pnl']:.2f}%")
    
    # 주간 리포트 스케줄
    next_report = notifier.schedule_weekly_report()
    print(f"✅ 다음 주간 리포트: {next_report.strftime('%Y-%m-%d %H:%M')}")
    
    return True


# ==========================================
# 메인 테스트 실행
# ==========================================

def main():
    print("\n" + "="*60)
    print("  AI 하이브리드 트레이딩 시스템 - 전체 테스트")
    print("="*60)
    
    tests = [
        ("시그널 감지", test_signal_detector),
        ("볼린저밴드", test_bollinger_bands),
        ("피보나치", test_fibonacci),
        ("15분봉 전략", test_strategy_15m),
        ("1시간봉 전략", test_strategy_1h),
        ("포지션 관리자", test_position_manager),
        ("리스크 관리자", test_risk_manager),
        ("Discord 알림", test_discord_notifier),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # 결과 요약
    print("\n" + "="*60)
    print("  테스트 결과 요약")
    print("="*60)
    
    passed = 0
    failed = 0
    for name, success, error in results:
        if success:
            print(f"  ✅ {name}: 통과")
            passed += 1
        else:
            print(f"  ❌ {name}: 실패 - {error}")
            failed += 1
    
    print(f"\n  총 {len(tests)}개 테스트: {passed}개 통과, {failed}개 실패")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
