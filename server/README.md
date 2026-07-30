# 추론 백엔드 (gray 서버)

브라우저가 붙는 게이트웨이 하나와, 모델별 GPU 워커로 구성됩니다.

```
브라우저 ──wss──▶ cloudflared ──▶ gateway :8080 ──┬─▶ oasis         :8000  (GPU 0)
                                  인증·대기열·TTL   ├─▶ diamond-atari :8000  (GPU 1)
                                                    ├─▶ diamond-csgo  :8000  (GPU 1)
                                                    └─▶ longlive      :8000  (GPU 2)
```

diamond-atari/diamond-csgo가 분리된 이유: DIAMOND는 GitHub 저장소의 `main`(Atari)과
`csgo` 브랜치가 아키텍처 자체가 달라(모듈 이름은 같지만 내부 구조가 다름) 한 프로세스에
같이 얹을 수 없습니다. 자세한 내용은 `workers/adapters/diamond_atari.py`와
`diamond_csgo.py`의 헤더 주석을 보세요.

게이트웨이만 터널에 노출됩니다. 워커 포트는 **절대 외부에 열지 마세요** — 인증이 없습니다.

---

## 1. 5분 안에 띄워보기 (가중치 없이)

체크포인트를 받기 전에 터널·TLS·CORS·대기열이 제대로 붙는지부터 확인하는 걸 권합니다.
더미 모델이 액션에 반응하는 절차적 영상을 생성하므로 파이프라인 전체를 끝까지 검증할 수 있습니다.

```bash
cp .env.example .env
sed -i "s/^WM_TOKENS=.*/WM_TOKENS=$(openssl rand -hex 24)/" .env
sed -i "s|^WM_ALLOWED_ORIGINS=.*|WM_ALLOWED_ORIGINS=https://<username>.github.io|" .env

./run_local.sh dummy          # Docker 없이. venv 만들고 워커 4개(oasis/diamond-atari/
                               # diamond-csgo/longlive) + 게이트웨이 기동
```

그다음 GitHub Pages 사이트를 열고 `.env`의 `WM_TOKENS` 값과 터널 주소를 입력하면 됩니다.
터널을 아직 안 붙였다면 로컬에서 `http://127.0.0.1:8080`으로 먼저 확인해도 됩니다
(단, 로컬 파일로 연 페이지는 `ws://`가 허용되지만 GitHub Pages는 `wss://`만 됩니다).

Docker를 쓸 수 있으면:

```bash
WM_DUMMY=1 docker compose up --build
```

> 학교 서버는 `docker` 그룹 권한이 없는 경우가 흔합니다. 그럴 때 `run_local.sh`를 쓰세요.
> 단점은 Oasis와 DIAMOND가 같은 venv를 공유한다는 것이라, torch 버전이 충돌하면
> 모델별로 venv를 나눠야 합니다 (`VENV=.venv-oasis ./run_local.sh` 식으로).

## 2. 테스트

```bash
python tests/test_stack.py
```

게이트웨이와 워커를 실제로 기동해 브라우저처럼 붙습니다. GPU도 가중치도 필요 없습니다.
프론트엔드가 의존하는 계약(close code, 대기열 승격, 프레임 헤더, `ack_seq` 반환, TTL)을 전부 검증합니다.

---

## 3. 실제 모델 연결

네 어댑터 모두 구현이 끝난 상태입니다 — 체크포인트만 받아서 경로를 채우면 됩니다.
추론 루프, 인코딩, 페이싱, 프레임 패킹은 공통 런타임이 처리합니다.

`workers/requirements-common.txt`는 게이트웨이·워커 공통 의존성(fastapi, numpy, ...)만
담고 있습니다. `torch`, `hydra-core`(DIAMOND), `gymnasium`+`ale-py`(DIAMOND-Atari — 직접
쓰진 않지만 `agent.py`가 무조건 `envs`를 import해서 없으면 로딩 자체가 실패합니다),
`omegaconf`/`peft`/`flash-attn`(LongLive)
같은 모델별 의존성은 각 저장소의 `requirements.txt`에서 설치하세요 — 이미 torch가 깔린
conda 환경을 워커별로 쓰는 걸 권장합니다(`WM_USE_VENV=0`, 위 1번 절 참고).

| 파일 | 내용 | 상태 |
|---|---|---|
| `workers/adapters/oasis.py` | 모델·VAE 로드 → 컨텍스트 초기화 → 잠재 샘플링 후 디코딩 | ✅ 실제 체크포인트로 동작 확인됨 |
| `workers/adapters/diamond_atari.py` | denoiser 로드 → 정적 시작 프레임 버퍼 → `sampler.sample()` | ✅ 구현 완료, 체크포인트 필요 |
| `workers/adapters/diamond_csgo.py` | `Agent`(denoiser+upsampler) + `WorldModelEnv` 로드 → spawn 데이터로 워밍업 | ✅ 구현 완료, 체크포인트+spawn 데이터+별도 저장소(csgo 브랜치) 필요 |
| `workers/adapters/longlive.py` | LongLive-1.3B 파이프라인을 블록 단위 스트리밍으로 래핑 | ✅ 구현 완료, 체크포인트 필요 |

### DIAMOND — Atari와 CS:GO는 서로 다른 저장소 브랜치

eloialonso/diamond의 `main`(Atari)과 `csgo` 브랜치는 코드가 통째로 다릅니다
(denoiser 1단계 vs base+upsampler 2단계, 액션 인코딩도 이산 vs 51차원 연속). 그래서:

* `models/diamond-atari` (현재 `main` 브랜치, `git worktree`로 관리) → `diamond-atari`가 사용.
* CS:GO를 쓰려면 **별도 디렉터리**에 csgo 브랜치를 따로 clone(또는 worktree) 해야 합니다:
  ```bash
  git clone -b csgo https://github.com/eloialonso/diamond models/diamond-csgo
  ```
  체크포인트와 함께 초기 컨텍스트용 spawn 데이터셋(실제 녹화 프레임)도 필요합니다:
  ```bash
  hf download eloialonso/diamond --include "csgo/*" \
      --local-dir models/diamond-csgo/downloads
  # csgo/model/csgo.pt  → WM_DIAMOND_CSGO_CKPT
  # csgo/spawn/         → WM_DIAMOND_CSGO_SPAWN_DIR
  ```
* Atari는 `atari_100k` 벤치마크 26개 게임 체크포인트를 **한 번에 전부** 받아서 한 워커가
  `model_id`로 골라 서빙합니다(게임마다 프로세스를 따로 안 띄웁니다):
  ```bash
  hf download eloialonso/diamond --include "atari_100k/models/*" \
      --local-dir server/weights/diamond_atari
  # server/weights/diamond_atari/atari_100k/models/*.pt 로 받아지니,
  # atari_100k/models/ 안의 *.pt 를 server/weights/diamond_atari/ 바로 아래로 옮기세요.
  ```
  `.env`에 `WM_DIAMOND_ATARI_WEIGHTS_DIR=server/weights/diamond_atari` 로 지정하면
  `diamond-atari-<게임소문자>`(예: `diamond-atari-breakout`, `diamond-atari-mspacman`)
  model_id로 각각 접속할 수 있습니다. 게임마다 학습된 액션 개수가 달라서(Breakout 4개,
  Alien 18개 등 — ALE 축소 액션셋) `workers/common/actions.py`의 `ATARI_GAME_ACTIONS`에
  게임별 정확한 목록을 `ale-py`로 직접 조회해 박아뒀습니다. 새로 맞출 값은 없습니다.

### LongLive — 반드시 `v1.0` 브랜치

`models/LongLive`의 기본(`main`) 브랜치는 **LongLive 2.0**(5B, 오프라인 배치 생성
전용 — 프레임 스트리밍이 안 됩니다)입니다. 이 프로젝트가 필요로 하는 실시간 인터랙티브
버전(LongLive-1.3B, `interactive_inference.py`)은 `v1.0` 브랜치에만 있습니다 —
이미 이 저장소에서 `git checkout v1.0`으로 전환해뒀습니다.

가중치가 **두 종류** 따로 필요합니다 (LongLive 체크포인트 하나만으론 안 됩니다):

```bash
# 1) Wan2.1-T2V-1.3B 베이스 (T5 텍스트 인코더 + VAE + 토크나이저) — LongLive가 아니라
#    원본 Wan2.1 저장소 것입니다. LongLive 코드가 이 경로를 상대경로로 하드코딩해서 찾습니다.
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir models/LongLive/wan_models/Wan2.1-T2V-1.3B

# 2) LongLive-1.3B 체크포인트 (generator + LoRA)
hf download Efficient-Large-Model/LongLive --local-dir models/LongLive/longlive_models
```

LongLive는 WASD로 조작하는 게임형 월드모델이 아니라 텍스트 프롬프트로 다음 장면을 계속
지시하는 롱비디오 생성기입니다. hotbar 슬롯(1~9)이 `WM_LONGLIVE_PROMPTS`(`|` 구분)의
프롬프트를 선택하는 스위치로 재활용됩니다 — 4번 절 참고.

구현해야 하는 인터페이스는 두 개뿐입니다 (`workers/common/base.py`):

```python
class WorldModel(ABC):
    fps: int = 20
    quality: int = 80

    def reset(self) -> np.ndarray: ...          # 첫 프레임
    def step(self, action: Action) -> np.ndarray: ...   # 다음 프레임
```

두 메서드는 **블로킹이어도 됩니다.** 공통 런타임이 세션별 스레드에서 호출하므로
이벤트 루프를 막지 않습니다. 반환 프레임은 HWC/CHW, uint8/float(`[0,1]` 또는 `[-1,1]`)
아무거나 괜찮습니다 — `encode.to_uint8_rgb()`가 알아서 정규화합니다.

Dockerfile에 각 리포를 clone하는 자리를 주석으로 표시해뒀습니다.

### 실시간성 확보

가장 큰 레버는 **diffusion step 수**입니다. 학습 때 50 step이었어도 추론은 4~8 step으로
줄여야 20FPS가 나옵니다. `WM_DIFFUSION_STEPS`(Oasis) / `WM_DENOISE_STEPS`(DIAMOND)로 조절하세요.

그다음이 fp16/bf16 autocast와 `channels_last`, 그다음이 컨텍스트 길이 축소입니다.
`torch.compile`은 첫 호출이 수십 초 걸리므로 기동 시 워밍업으로 한 번 돌려두세요.

---

## 4. 액션 매핑

프론트엔드는 `KeyboardEvent.code`를 그대로 보냅니다("KeyW", "ShiftLeft"...).
모델 액션 스페이스로의 변환은 전부 `workers/common/actions.py`에 있습니다.

| 모델 | 함수 | 출력 |
|---|---|---|
| Oasis | `oasis_vector()` | float32[25] — open-oasis `ACTION_KEYS` 규약 |
| DIAMOND CS:GO | `csgo_vector()` | float32[51] — 키 11 + 발사/조준 2 + 마우스 원핫(23+15) |
| DIAMOND Atari | `to_atari()` | int — ALE 표준 18개 중 하나 |
| LongLive | `longlive_prompt_index()` | int — hotbar(1~9) → `WM_LONGLIVE_PROMPTS` 인덱스 |

**카메라 delta는 ±20도로 클램프됩니다.** 한 틱에 그보다 크게 도는 입력은 학습 분포 밖이라
모델 출력이 무너집니다. CS:GO의 `CSGO_MOUSE_X_BINS`/`CSGO_MOUSE_Y_BINS`·키 순서는 원 리포
`src/csgo/action_processing.py`의 `encode_csgo_action()`과 정확히 맞춰뒀습니다(총 51차원 —
X/Y 축 bin 개수가 다르고 0을 포함합니다. 옛 62차원 벡터는 실제 체크포인트와 맞지 않던
버그였습니다).

LongLive는 WASD가 아니라 hotbar 숫자로 "지금 재생 중인 프롬프트"를 고릅니다 — 3번 절의
LongLive 설명 참고.

새 모델을 추가하려면 이 파일에 매퍼 함수 하나만 더 쓰면 됩니다.

---

## 5. 세션 관리

GPU 1장은 동시 세션 1개가 안전합니다. 게이트웨이가 이걸 강제합니다.

| 환경변수 | 기본 | 설명 |
|---|---|---|
| `WM_CAP_*` | 1 | 모델별 동시 세션 수 |
| `WM_SESSION_TTL` | 120 | 세션당 최대 초. 초과 시 close `4008` |
| `WM_IDLE_TIMEOUT` | 30 | 입력이 없으면 회수 (탭 켜두고 자리 비우는 경우) |
| `WM_MAX_QUEUE` | 12 | 대기열 상한. 초과 시 close `4029` |

대기 중인 클라이언트에게는 순번이 바뀔 때마다 `{"type":"queue","position":n,"ahead":n-1,"eta":초}`가
전송되고, 앞 세션이 끝나면 자동으로 승격됩니다. 프론트엔드는 이걸 이미 처리하고 있어서
서버에서 보내주기만 하면 대기열 화면이 뜹니다.

`GET /models`로 현재 부하를 볼 수 있습니다.

```json
{"models":[{"id":"oasis","capacity":1,"active":1,"queued":3,"eta":480}], "session_ttl":120}
```

`WM_CAP_OASIS` / `WM_CAP_DIAMOND_ATARI` / `WM_CAP_DIAMOND_CSGO` / `WM_CAP_LONGLIVE` 로
모델별 동시 세션 수를 각각 조절합니다.

---

## 6. 접근 제어

프론트엔드의 키 입력창은 UX용이지 보안이 아닙니다. 실제 차단은 두 겹입니다.

**(a) 게이트웨이 토큰** — `WM_TOKENS`에 쉼표로 나열, `hmac.compare_digest`로 상수시간 비교.
실패 시 close `4001`. 랩 슬랙에 공유하는 단일 키로 시작하고, 사람별 추적이 필요해지면
토큰을 여러 개 발급해 누가 어느 걸 쓰는지 기록하면 됩니다.

> `WM_TOKENS`를 비워두면 **인증 없이** 동작합니다(개발 편의). 배포 전 반드시 채우세요.
> 로그에 경고가 찍힙니다.

**(b) Cloudflare Access** — 터널 앞단에서 랩 이메일 도메인만 통과시킵니다. 무료 50명.

```
Cloudflare Zero Trust → Access → Applications → Self-hosted
  Domain: wm.gray.example.dev
  Policy: Allow · Emails ending in @<university>.ac.kr
```

WebSocket에는 브라우저가 커스텀 헤더를 못 붙이므로, Access 쿠키가 자동 전달되도록
**터널 도메인과 Access 애플리케이션 도메인을 동일하게** 두세요.

---

## 7. 터널

```bash
cloudflared tunnel login
cloudflared tunnel create wm
cloudflared tunnel route dns wm wm.gray.example.dev

cat > ~/.cloudflared/config.yml <<'YAML'
tunnel: wm
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: wm.gray.example.dev
    service: http://localhost:8080
  - service: http_status:404
YAML

cloudflared tunnel run wm
```

아웃바운드 연결만 쓰므로 랩 방화벽의 인바운드 정책을 건드릴 필요가 없고, TLS가 자동으로
붙어서 `wss://`가 그냥 됩니다. systemd 서비스로 등록해두면 재부팅에도 살아납니다
(`cloudflared service install`).

Docker로 띄운다면 `.env`에 `CF_TUNNEL_TOKEN`을 넣고:

```bash
docker compose --profile tunnel up -d
```

---

## 8. 참고 수치

더미 모델 기준 (렌더링 + JPEG 인코딩, CPU만):

| 항목 | 값 |
|---|---|
| 렌더+인코딩 | 4.2 ms/frame (약 240 fps 여유) |
| JPEG 크기 | 320×180 q80 → 약 7.8 KB |
| 대역폭 | 약 1.3 Mbps/세션 @ 20fps |

실제 모델에서는 추론이 병목이라 이 값들은 상한으로만 보세요.
해상도를 2배로 올리면 대역폭은 대략 4배가 됩니다.

지연은 `추론 시간 + 인코딩 + 네트워크 RTT`입니다. 같은 캠퍼스 네트워크면 RTT 20~40ms,
외부에서 접속하면 60~120ms 정도를 예상하세요. 브라우저 HUD의 지연 표시는 `ack_seq`
왕복으로 실측한 값이라, 튜닝할 때 이 숫자를 보면서 조절하면 됩니다.

---

## 9. 디렉터리

```
server/
├── gateway/app.py                인증 · 대기열 · TTL · 워커 프록시
├── workers/
│   ├── run.py                    워커 진입점 (WM_MODEL로 어댑터 선택)
│   ├── common/
│   │   ├── base.py               WorldModel 인터페이스 — 새 모델은 여기만 보면 됨
│   │   ├── actions.py            KeyboardEvent.code → 모델 액션 스페이스
│   │   ├── encode.py             프레임 정규화 · JPEG · 와이어 패킹
│   │   └── server.py             추론 루프 (최신 액션 우선, 페이싱)
│   └── adapters/
│       ├── oasis.py              open-oasis, 실제 체크포인트로 동작 확인됨
│       ├── diamond_atari.py      DIAMOND main 브랜치(Atari)
│       ├── diamond_csgo.py       DIAMOND csgo 브랜치(CS:GO) — 별도 저장소 체크아웃 필요
│       ├── longlive.py           LongLive v1.0(1.3B) — 텍스트 프롬프트 기반 롱비디오
│       └── dummy.py              가중치 없이 파이프라인 검증용
├── tests/test_stack.py           통합 테스트
├── docker-compose.yml            모델별 GPU 할당
├── run_local.sh                  Docker 없이 실행
└── .env.example
```
