#!/usr/bin/env bash
# Docker 없이 gray 서버에서 바로 띄우기.
#
# 학교 서버는 docker 그룹 권한이 없는 경우가 많습니다. 그럴 때 이 스크립트로
# venv 하나에 전부 올려서 씁니다. 단점은 Oasis와 DIAMOND가 같은 파이썬 환경을
# 공유한다는 것이라, torch 버전이 충돌하면 venv를 모델별로 나눠야 합니다.
#
#   ./run_local.sh dummy     # 가중치 없이 파이프라인만
#   ./run_local.sh real      # 실제 체크포인트로
#   ./run_local.sh stop

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-dummy}"
VENV="${VENV:-.venv}"
RUN_DIR=".run"
mkdir -p "$RUN_DIR"

stop_all() {
  for f in "$RUN_DIR"/*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      echo "중지: $(basename "$f" .pid) (pid $pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  done
}

if [ "$MODE" = "stop" ]; then
  stop_all
  exit 0
fi

# --- 환경 준비 -------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "venv 생성: $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r gateway/requirements.txt -r workers/requirements-common.txt

if [ -f .env ]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
else
  echo "경고: .env 없음 — .env.example을 복사해 채우세요. 인증 없이 뜹니다."
fi

if [ "$MODE" = "dummy" ]; then
  export WM_DUMMY=1
  echo "== 더미 모드 (실제 추론 아님) =="
else
  export WM_DUMMY=0
fi

stop_all
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- 워커 -----------------------------------------------------------------
launch() {  # launch <이름> <포트> <WM_MODEL> <기본모델> <GPU>
  local name=$1 port=$2 adapter=$3 default=$4 gpu=$5
  echo "기동: $name  포트 $port  GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WM_MODEL="$adapter" WM_DEFAULT_MODEL="$default" WM_PORT="$port" \
    python -m workers.run > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
}

launch oasis   8001 oasis   oasis        "${GPU_OASIS:-0}"
launch diamond 8002 diamond diamond-csgo "${GPU_DIAMOND:-1}"

# --- 게이트웨이 ------------------------------------------------------------
export WM_WORKER_OASIS="ws://127.0.0.1:8001/session"
export WM_WORKER_DIAMOND_CSGO="ws://127.0.0.1:8002/session"
export WM_WORKER_DIAMOND_ATARI="ws://127.0.0.1:8002/session"

echo "기동: gateway 포트 8080"
uvicorn gateway.app:app --host 127.0.0.1 --port 8080 \
  --ws-max-size 16777216 --no-access-log > "$RUN_DIR/gateway.log" 2>&1 &
echo $! > "$RUN_DIR/gateway.pid"

sleep 3
echo
if curl -fsS http://127.0.0.1:8080/healthz > /dev/null; then
  echo "✓ 게이트웨이 정상: http://127.0.0.1:8080"
  curl -fsS http://127.0.0.1:8080/models
  echo
else
  echo "✗ 게이트웨이 기동 실패 — $RUN_DIR/gateway.log 확인"
  exit 1
fi

cat <<'EOF'

다음 단계:
  1) 터널 연결   cloudflared tunnel run wm
  2) 브라우저에서 GitHub Pages 사이트 → 서버 주소에 wss://<터널도메인> 입력
  로그      tail -f .run/*.log
  중지      ./run_local.sh stop
EOF
