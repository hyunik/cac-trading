# 대시보드 시각화 및 배포 설정

## 1. UI 구성
- **Real-time Signal Feed**: 15m/1H 전략에서 감지된 시그널 캔들을 실시간 텍스트 및 로그로 출력.
- **Comparison Chart**: 두 전략의 누적 수익 곡선을 대조 표시.
- **AI Analytics**: LLM이 판단한 '진입 보정 근거'와 '심리적 원칙 준수도' 점수화.

## 2. Netlify 배포
- **Model Selector**: ChatGPT-4o와 Gemini 1.5 Pro 중 사용자가 엔진 선택.
- **Environment**: Binance API 및 LLM API Key 보안 설정.
- **Review Loop**: 매매 종료 후 "왜 원칙대로 손절하지 않았나?" [cite_start]등의 AI 잔소리 기능 활성화[cite: 34].