# 프로젝트명: AI 하이브리드 코인 선물 페이퍼 트레이딩 시스템

## 1. 프로젝트 목적
- **비교 분석**: 15분봉 추세 추종 매매법과 1시간봉 볼린저밴드 역추세 매매법의 성과를 실시간으로 대조함.
- **AI 지능화**: ChatGPT/Gemini를 활용하여 차트의 수치적 지표가 부족한 2%를 메우는 '진입 보정' 엔진 구현.
- **리스크 관리**: 엄격한 손절 원칙과 피보나치 기반의 복구 알고리즘을 통해 심리적 요인을 배제한 기계적 매매 실현.

## 2. 핵심 기술 스택
- **Analysis Engine**: Python (Pandas, Ta-Lib) + LLM (ChatGPT-4o / Gemini 1.5 Pro)
- **Frontend**: React + TradingView Lightweight Charts
- **Deployment**: Netlify (Frontend & Functions)
- **Data Source**: Binance API (Real-time WebSocket)