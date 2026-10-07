#!/usr/bin/env bash
# VideoWorldModel server and Cloudflare tunnel launcher.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SERVER_DIR="$PROJECT_ROOT/server"
CLOUDFLARED="$PROJECT_ROOT/bin/cloudflared"
RUN_LOCAL="$SERVER_DIR/run_local.sh"

usage() {
  printf '%s\n' \
    'Usage:' \
    '  scripts/server.sh start [real|dummy] [--gpus 4,5,6,7]' \
    '  scripts/server.sh stop' \
    '  scripts/server.sh restart [real|dummy] [--gpus 4,5,6,7]' \
    '  scripts/server.sh status' \
    '  scripts/server.sh tunnel [url]' \
    '  scripts/server.sh up [real|dummy] [--gpus 4,5,6,7]' \
    '' \
    'GPU order: oasis, diamond-atari, diamond-csgo, longlive'
}

run_server() {
  local mode=${1:-real}
  shift || true
  case "$mode" in
    real|dummy) ;;
    *) echo "Unknown server mode: $mode" >&2; usage; exit 2 ;;
  esac

  local gpu_list=${VWM_SERVER_GPUS:-}
  while (($#)); do
    case "$1" in
      --gpus)
        [[ $# -ge 2 ]] || { echo '--gpus requires four comma-separated GPU IDs' >&2; exit 2; }
        gpu_list=$2
        shift 2
        ;;
      *) echo "Unknown server option: $1" >&2; usage; exit 2 ;;
    esac
  done

  if [[ -n "$gpu_list" ]]; then
    local gpu_oasis gpu_atari gpu_csgo gpu_longlive extra
    IFS=',' read -r gpu_oasis gpu_atari gpu_csgo gpu_longlive extra <<< "$gpu_list"
    if [[ -z "$gpu_oasis" || -z "$gpu_atari" || -z "$gpu_csgo" || -z "$gpu_longlive" || -n "${extra:-}" ]]; then
      echo "Expected exactly four GPU IDs: --gpus 4,5,6,7" >&2
      exit 2
    fi
    for gpu_id in "$gpu_oasis" "$gpu_atari" "$gpu_csgo" "$gpu_longlive"; do
      [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu_id" >&2; exit 2; }
    done
    export GPU_OASIS=$gpu_oasis
    export GPU_DIAMOND_ATARI=$gpu_atari
    export GPU_DIAMOND_CSGO=$gpu_csgo
    export GPU_LONGLIVE=$gpu_longlive
    echo "GPU assignment: oasis=$gpu_oasis atari=$gpu_atari csgo=$gpu_csgo longlive=$gpu_longlive"
  fi

  (
    cd "$SERVER_DIR"
    WM_USE_VENV="${WM_USE_VENV:-0}" bash "$RUN_LOCAL" "$mode"
  )
}

stop_server() {
  (
    cd "$SERVER_DIR"
    WM_USE_VENV="${WM_USE_VENV:-0}" bash "$RUN_LOCAL" stop
  )
}

show_status() {
  if curl -fsS --max-time 2 http://127.0.0.1:8080/healthz; then
    echo
    curl -fsS --max-time 2 http://127.0.0.1:8080/models || true
    echo
  else
    echo "VideoWorldModel server is not responding on http://127.0.0.1:8080" >&2
    return 1
  fi
}

run_tunnel() {
  local url=${1:-${VWM_TUNNEL_URL:-http://localhost:8080}}
  if [[ ! -x "$CLOUDFLARED" ]]; then
    echo "cloudflared executable not found: $CLOUDFLARED" >&2
    exit 1
  fi
  exec "$CLOUDFLARED" tunnel --url "$url"
}

command=${1:-help}
case "$command" in
  start)
    shift
    run_server "${1:-real}" "${@:2}"
    ;;
  stop)
    stop_server
    ;;
  restart)
    shift
    stop_server
    run_server "${1:-real}" "${@:2}"
    ;;
  status)
    show_status
    ;;
  tunnel)
    run_tunnel "${2:-}"
    ;;
  up)
    shift
    run_server "${1:-real}" "${@:2}"
    run_tunnel "${VWM_TUNNEL_URL:-http://localhost:8080}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage
    exit 2
    ;;
esac
