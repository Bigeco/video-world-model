# Web demo

브라우저에서 Oasis, DIAMOND(Atari/CS:GO), LongLive를 실시간으로 조작해 보는 playground입니다.
연구의 motivation을 보여주는 용도이며, 이 디렉터리만 GitHub `main`에 공개됩니다.

- `web/index.html`: 프론트엔드. 브라우저로 직접 열거나 GitHub Pages로 배포합니다.
- `server/`: 게이트웨이와 모델별 GPU 워커. 자세한 내용은 [server/README.md](server/README.md).
- `scripts/server.sh`: 서버 시작·재시작·상태·Cloudflare tunnel. [scripts/README.md](scripts/README.md).
- `results/server_sessions/`: 세션별 프레임과 지연 통계(커밋하지 않음).
- `results/playground/`: 이전 레이아웃에서 남은 세션 기록(커밋하지 않음).

```bash
cd demo
scripts/server.sh restart real --gpus 4,5,6,7
scripts/server.sh status
scripts/server.sh tunnel
```

실제 모델은 `server/.env`의 `VWM_MODEL_FACTORY`로 연결합니다. 워커는 기본적으로
`../research`를 `PYTHONPATH`에 추가하며, 다른 위치는 `VWM_RESEARCH_ROOT`로 지정합니다.
외부 모델 저장소는 `../external/models/`에서 찾습니다.

`scripts/server.sh stop`은 이 사용자의 `workers.run`/`gateway.app` 프로세스를 모두
정리하므로, 다른 checkout에서 띄운 서버도 함께 종료됩니다.
