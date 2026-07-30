# DIAMOND(CS:GO) 워커. `csgo` 브랜치 코드 — Atari(main 브랜치)와 아키텍처가 달라
# 별도 이미지/저장소 체크아웃이 필요하다 (diamond_csgo.py 헤더 주석 참고).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY workers/requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# ---------------------------------------------------------------------
# DIAMOND 리포 (csgo 브랜치):
#   RUN git clone -b csgo https://github.com/eloialonso/diamond /opt/diamond-csgo \
#    && pip install --no-cache-dir -r /opt/diamond-csgo/requirements.txt
#   ENV PYTHONPATH=/opt/diamond-csgo/src:$PYTHONPATH
#
# 체크포인트와 함께 spawn 데이터셋(초기 컨텍스트용 실제 녹화 프레임)도 받아야 한다:
#   RUN huggingface-cli download eloialonso/diamond --include "csgo/*" \
#         --local-dir /opt/diamond-csgo/downloads
#   csgo/model/csgo.pt → WM_DIAMOND_CSGO_CKPT, csgo/spawn/ → WM_DIAMOND_CSGO_SPAWN_DIR
# ---------------------------------------------------------------------

COPY workers/ /app/workers/

EXPOSE 8000
ENV WM_MODEL=diamond-csgo
CMD ["python", "-m", "workers.run"]
