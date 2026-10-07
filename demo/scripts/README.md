# 실행 스크립트

모든 명령은 `demo/` 디렉터리에서 실행하는 것을 기준으로 합니다.
스크립트가 프로젝트 루트를 자동으로 찾으므로 다른 디렉터리에서 절대경로로 호출해도
동작합니다.

## 빠른 시작

보통 사용할 수 있는 GPU가 4, 5, 6, 7이라면 다음 명령으로 실제 서버를 재시작합니다.

```bash
scripts/server.sh restart real --gpus 4,5,6,7
```

GPU 배정 순서는 고정되어 있습니다.

| 순서 | 모델 | 위 명령의 물리 GPU |
|---:|---|---:|
| 1 | Oasis | 4 |
| 2 | DIAMOND Atari | 5 |
| 3 | DIAMOND CS:GO | 6 |
| 4 | LongLive | 7 |

서버가 정상인지 확인한 다음 Cloudflare Quick Tunnel을 실행합니다.

```bash
scripts/server.sh status
scripts/server.sh tunnel
```

`tunnel`은 현재 터미널을 점유합니다. 종료하려면 `Ctrl+C`를 누릅니다. 터널을 종료해도
백그라운드 서버는 계속 실행됩니다.

## 서버 관리

### 시작

```bash
scripts/server.sh start real --gpus 4,5,6,7
```

가중치 없이 게이트웨이와 WebSocket 경로만 확인하려면 dummy 모드를 사용합니다.

```bash
scripts/server.sh start dummy --gpus 4,5,6,7
```

### 재시작

```bash
scripts/server.sh restart real --gpus 4,5,6,7
```

위 명령은 기존의 다음 명령을 대체합니다.

```bash
WM_USE_VENV=0 bash server/run_local.sh stop
WM_USE_VENV=0 bash server/run_local.sh real
```

`WM_USE_VENV`를 지정하지 않으면 `server.sh`는 기본적으로 현재 활성화된 conda/Python
환경을 사용하도록 `WM_USE_VENV=0`을 적용합니다. 서버 자체 venv를 사용하려면 다음과
같이 명시합니다.

```bash
WM_USE_VENV=1 scripts/server.sh restart real --gpus 4,5,6,7
```

### 상태 확인과 종료

```bash
scripts/server.sh status
scripts/server.sh stop
```

상태 확인은 게이트웨이의 `/healthz`와 `/models` 응답을 출력합니다.

### 서버와 터널을 한 번에 실행

```bash
scripts/server.sh up real --gpus 4,5,6,7
```

`up`은 서버를 시작한 뒤 현재 터미널에서 Quick Tunnel을 실행합니다.

## GPU 설정

### 서버 GPU 네 개 지정

```bash
scripts/server.sh restart real --gpus 4,5,6,7
```

동일한 GPU를 둘 이상의 워커에 배정할 수도 있습니다.

```bash
scripts/server.sh restart real --gpus 4,5,5,7
```

이 경우 DIAMOND Atari와 CS:GO가 물리 GPU 5를 공유합니다. 동시에 모델을 실행하면
메모리가 부족할 수 있으므로 용량을 확인해야 합니다.

자주 같은 GPU를 사용한다면 환경변수로 기본값을 지정할 수 있습니다.

```bash
export VWM_SERVER_GPUS=4,5,6,7
scripts/server.sh restart real
```

개별 환경변수를 직접 지정하는 기존 방식도 지원됩니다.

```bash
GPU_OASIS=4 \
GPU_DIAMOND_ATARI=5 \
GPU_DIAMOND_CSGO=6 \
GPU_LONGLIVE=7 \
scripts/server.sh restart real
```

## Cloudflare Tunnel

기본 주소 `http://localhost:8080`을 노출합니다.

```bash
scripts/server.sh tunnel
```

기존 명령과 동일합니다.

```bash
bin/cloudflared tunnel --url http://localhost:8080
```

다른 로컬 주소를 사용하려면 인자 또는 환경변수로 지정합니다.

```bash
scripts/server.sh tunnel http://localhost:8081

VWM_TUNNEL_URL=http://localhost:8081 scripts/server.sh tunnel
```

## 도움말

```bash
scripts/server.sh --help
```
