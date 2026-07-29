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
  local f pid n
  # 1차: PID 파일 기준으로 정중하게 종료 (TERM → 최대 10초 대기 → KILL)
  for f in "$RUN_DIR"/*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "중지: $(basename "$f" .pid) (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for n in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "  → 응답 없음, 강제 종료"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$f"
  done

  # 2차: PID 파일이 유실된 고아 프로세스 정리.
  # GPU 메모리는 워커 프로세스가 죽어야 풀리므로 이 단계가 중요하다.
  # 내 소유 프로세스만, 이 프로젝트의 명령줄 패턴에만 해당한다.
  for pat in 'workers\.run' 'uvicorn gateway\.app'; do
    pids=$(pgrep -u "$(id -u)" -f "$pat" 2>/dev/null || true)
    for pid in $pids; do
      [ "$pid" = "$$" ] && continue
      echo "고아 프로세스 정리: pid $pid ($pat)"
      kill "$pid" 2>/dev/null || true
      sleep 0.5
      kill -9 "$pid" 2>/dev/null || true
    done
  done

  # GPU가 실제로 비었는지 알려준다 (nvidia-smi가 있을 때만)
  if command -v nvidia-smi > /dev/null 2>&1; then
    local mine
    mine=$(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader 2>/dev/null || true)
    if [ -n "$mine" ]; then
      echo
      echo "아직 GPU를 쓰는 프로세스가 남아 있습니다:"
      echo "$mine" | sed 's/^/  /'
      echo "  (다른 작업일 수 있습니다. ps -fp <PID> 로 확인하세요)"
    fi
  fi
}

if [ "$MODE" = "stop" ]; then
  stop_all
  exit 0
fi

# --- 파이썬 환경 준비 ------------------------------------------------------
# 기본은 로컬 venv 를 만들어 격리한다. 하지만 이미 conda 등 원하는 환경을
# 활성화해 두었다면 WM_USE_VENV=0 으로 그걸 그대로 쓴다.
#   (예: conda activate myenv && WM_USE_VENV=0 bash run_local.sh real)
# 실제 모델(torch, open-oasis)을 돌릴 때는 이미 torch 가 깔린 conda 환경을
# 쓰는 편이 편하므로 이 옵션을 권장한다.
USE_VENV="${WM_USE_VENV:-1}"

if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV" ]; then
    echo "venv 생성: $VENV"
    # --system-site-packages 를 주면 conda/시스템의 torch 를 그대로 본다.
    python3 -m venv "$VENV" ${WM_VENV_SYSTEM_SITE:+--system-site-packages}
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# 실행에 쓸 파이썬을 결정한다. 'python' 이 없는 환경(일부 conda)도 있으므로
# python → python3 순으로 찾고, 이후 전부 이 PY 변수로 호출한다.
if command -v python > /dev/null 2>&1; then
  PY=python
elif command -v python3 > /dev/null 2>&1; then
  PY=python3
else
  echo "✗ python 도 python3 도 없습니다. conda 환경을 activate 했는지 확인하세요."
  exit 1
fi
echo "파이썬: $(command -v $PY) ($($PY --version 2>&1))"

# pip 도 활성 환경 것을 확실히 쓰도록 '$PY -m pip' 로 호출한다.
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r gateway/requirements.txt -r workers/requirements-common.txt

# .env 로드.
# `source .env`를 쓰지 않는다 — 값에 <, >, $, & 같은 문자가 있으면 셸이
# 리다이렉션이나 변수 확장으로 해석해버린다. 한 줄씩 직접 파싱한다.
load_env() {
  local file=$1 line key val
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}                                  # CRLF 제거
    case "$line" in ''|'#'*) continue ;; esac
    [ "${line#*=}" = "$line" ] && continue              # '=' 없는 줄 무시
    key=${line%%=*}
    val=${line#*=}
    key=$(printf '%s' "$key" | tr -d '[:space:]')
    case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac  # 이상한 키 무시
    val=${val#"${val%%[![:space:]]*}"}                   # 앞쪽 공백 제거
    case "$val" in
      # 따옴표로 감싼 값: 닫는 따옴표까지가 값, 그 뒤는 주석으로 버린다
      \"*) val=${val#\"}; val=${val%%\"*} ;;
      \'*) val=${val#\'}; val=${val%%\'*} ;;
      # 맨값: ' #' 또는 '<TAB>#' 이후는 인라인 주석
      *)
        case "$val" in *" #"*)     val=${val%% #*} ;; esac
        case "$val" in *"$(printf '\t')#"*) val=${val%%"$(printf '\t')"#*} ;; esac
        val=${val%"${val##*[![:space:]]}"}               # 뒤쪽 공백 제거
        ;;
    esac
    export "$key=$val"
  done < "$file"
}

if [ -f .env ]; then
  load_env .env
  if printf '%s' "${WM_ALLOWED_ORIGINS:-}" | grep -q '[<>]'; then
    echo "경고: WM_ALLOWED_ORIGINS에 <꺾쇠>가 남아 있습니다 — 실제 값으로 바꾸세요."
    echo "       현재: $WM_ALLOWED_ORIGINS"
  fi
  if [ -z "${WM_TOKENS:-}" ] || [ "${WM_TOKENS}" = "changeme-run-openssl-rand-hex-24" ]; then
    echo "경고: WM_TOKENS가 기본값입니다 — 인증 없이 뜹니다."
    echo "       openssl rand -hex 24 로 생성해 .env에 넣으세요."
  fi
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
export PYTHONUNBUFFERED=1        # 로그가 즉시 파일에 찍히도록

# 유저 site-packages(~/.local)를 무시한다.
# 여기에 활성 환경과 다른 버전의 torch 등이 깔려 있으면 그게 섞여 들어와
# "드라이버가 너무 오래됨"(cu130 vs 드라이버) 같은 충돌을 일으킨다.
# 이걸 켜면 워커는 오직 현재 환경(conda/venv)의 패키지만 쓴다.
# 끄려면 WM_ALLOW_USERSITE=1 로 실행.
if [ "${WM_ALLOW_USERSITE:-0}" != "1" ]; then
  export PYTHONNOUSERSITE=1
fi

# 포트가 뜰 때까지 대기. 모델 로딩이 오래 걸릴 수 있어 넉넉히 준다.
wait_up() {  # wait_up <이름> <포트> <최대초>
  local name=$1 port=$2 limit=${3:-60} i
  for i in $(seq 1 $((limit * 2))); do
    if curl -fsS -m 2 "http://127.0.0.1:$port/healthz" > /dev/null 2>&1; then
      return 0
    fi
    # 프로세스가 죽었으면 더 기다릴 이유가 없다
    if [ -f "$RUN_DIR/$name.pid" ] && ! kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null; then
      return 2
    fi
    sleep 0.5
  done
  return 1
}

report_fail() {  # report_fail <이름>
  echo "✗ $1 기동 실패. 마지막 로그:"
  echo "------------------------------------------------------------"
  tail -n 25 "$RUN_DIR/$1.log" 2>/dev/null || echo "(로그 없음)"
  echo "------------------------------------------------------------"
}

# --- 워커 -----------------------------------------------------------------
launch() {  # launch <이름> <포트> <WM_MODEL> <기본모델> <GPU>
  local name=$1 port=$2 adapter=$3 default=$4 gpu=$5
  echo "기동: $name  포트 $port  GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WM_MODEL="$adapter" WM_DEFAULT_MODEL="$default" WM_PORT="$port" \
    "$PY" -m workers.run > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
}

launch oasis   8001 oasis   oasis        "${GPU_OASIS:-0}"
launch diamond 8002 diamond diamond-csgo "${GPU_DIAMOND:-1}"

# --- 게이트웨이 ------------------------------------------------------------
export WM_WORKER_OASIS="ws://127.0.0.1:8001/session"
export WM_WORKER_DIAMOND_CSGO="ws://127.0.0.1:8002/session"
export WM_WORKER_DIAMOND_ATARI="ws://127.0.0.1:8002/session"

echo "기동: gateway 포트 8080"
# `uvicorn` 콘솔 스크립트 대신 `python -m`을 쓴다.
# conda + venv가 겹친 환경에서는 스크립트가 PATH에 없을 수 있다.
"$PY" -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 \
  --ws-max-size 16777216 --no-access-log > "$RUN_DIR/gateway.log" 2>&1 &
echo $! > "$RUN_DIR/gateway.pid"

# --- 기동 확인 ------------------------------------------------------------
# 워커는 체크포인트 로딩 때문에 오래 걸릴 수 있다. 더미 모드면 금방 뜬다.
WAIT_LIMIT=${WM_STARTUP_TIMEOUT:-$([ "$MODE" = "dummy" ] && echo 60 || echo 300)}
echo
echo "기동 대기 중 (최대 ${WAIT_LIMIT}초)…"

failed=0
for entry in "oasis 8001" "diamond 8002" "gateway 8080"; do
  set -- $entry
  name=$1 port=$2
  wait_up "$name" "$port" "$WAIT_LIMIT"
  case $? in
    0) echo "✓ $name 정상 (:$port)" ;;
    2) echo "✗ $name 프로세스가 종료됨"; report_fail "$name"; failed=1 ;;
    *) echo "✗ $name 응답 없음 (${WAIT_LIMIT}초 초과)"; report_fail "$name"; failed=1 ;;
  esac
done

if [ "$failed" = "1" ]; then
  stop_all
  exit 1
fi

echo
curl -fsS http://127.0.0.1:8080/models
echo

cat <<'EOF'

다음 단계:
  1) 터널 연결   cloudflared tunnel run wm
  2) 브라우저에서 GitHub Pages 사이트 → 서버 주소에 wss://<터널도메인> 입력
  로그      tail -f .run/*.log
  중지      ./run_local.sh stop
EOF
