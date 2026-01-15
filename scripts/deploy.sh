#!/bin/bash
# Vultr 서버 배포 스크립트

set -e

echo "=== CAC Trading System 배포 ==="

# 1. 시스템 패키지 업데이트
echo "[1/5] 시스템 업데이트..."
sudo apt update && sudo apt upgrade -y

# 2. Python 설치
echo "[2/5] Python 설치..."
sudo apt install -y python3 python3-pip python3-venv git

# 3. 프로젝트 클론 (또는 pull)
echo "[3/5] 프로젝트 클론..."
PROJECT_DIR="/home/$USER/cac-trading"

if [ -d "$PROJECT_DIR" ]; then
    cd $PROJECT_DIR
    git pull origin main
else
    git clone https://github.com/hyunik/cac-trading.git $PROJECT_DIR
    cd $PROJECT_DIR
fi

# 4. 가상환경 및 의존성 설치
echo "[4/5] 의존성 설치..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. 환경변수 설정 확인
echo "[5/5] 환경변수 확인..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  .env 파일을 수정해주세요!"
    echo "   nano $PROJECT_DIR/.env"
fi

echo ""
echo "=== 배포 완료 ==="
echo ""
echo "실행 명령어:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python -m src.main"
echo ""
echo "백그라운드 실행 (systemd):"
echo "  sudo systemctl start cac-trading"
echo "  sudo systemctl enable cac-trading"
