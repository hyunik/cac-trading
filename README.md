# AI 하이브리드 코인 선물 페이퍼 트레이딩 시스템

15분봉 추세 피라미딩 전략과 1시간봉 볼린저밴드 역추세 전략을 비교 분석하는 자동화 트레이딩 시스템

## 기능

- **전략 A (15분봉)**: 일봉 바이어스 → 피라미딩 (10%→20%→60%)
- **전략 B (1시간봉)**: 볼린저밴드 터치 → 3단계 익절
- **Discord 알림**: 실시간 진입/익절/손절 + 주간 리포트
- **리스크 관리**: 일일/주간 손실 한도, 연속 손실 제한

## 대상 코인 (11개)

| 코인 | 심볼 |
|------|------|
| Bitcoin | BTCUSDT |
| Ethereum | ETHUSDT |
| Solana | SOLUSDT |
| XRP | XRPUSDT |
| Cardano | ADAUSDT |
| Aave | AAVEUSDT |
| PEPE | 1000PEPEUSDT |
| Optimism | OPUSDT |
| Arbitrum | ARBUSDT |
| dogwifhat | WIFUSDT |
| Jito | JTOUSDT |

## 빠른 시작

```bash
# 1. 클론
git clone https://github.com/YOUR_USERNAME/cac-trading.git
cd cac-trading

# 2. 가상환경 설정
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일에 DISCORD_WEBHOOK_URL 입력

# 5. 실행
python -m src.main
```

## 프로젝트 구조

```
src/
├── analysis/        # 시그널 감지, 볼린저밴드, 피보나치
├── strategies/      # 15분봉/1시간봉 전략
├── data/            # Binance API
├── core/            # 포지션/리스크 관리
├── notifications/   # Discord 알림
└── main.py          # 메인 시스템
```

## 환경변수

| 변수 | 설명 |
|------|------|
| `TRADING_SYMBOL` | 거래 심볼 (기본: BTCUSDT) |
| `INITIAL_CAPITAL` | 초기 자본 (기본: 10000) |
| `DISCORD_WEBHOOK_URL` | Discord Webhook URL |

## 라이선스

MIT
