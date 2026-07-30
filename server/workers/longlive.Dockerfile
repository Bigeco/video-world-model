# LongLive(v1.0, LongLive-1.3B) 워커. torch/CUDA 버전이 oasis/diamond와 달라
# (LongLive 문서 기준 torch==2.8.0+cu128) 별도 이미지로 분리한다.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY workers/requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# ---------------------------------------------------------------------
# LongLive 리포 — 반드시 v1.0 브랜치(main은 LongLive 2.0/5B 오프라인 배치 파이프라인이라
# 이 프로젝트의 실시간 스트리밍 어댑터(longlive.py)와 맞지 않는다):
#   RUN git clone --single-branch -b v1.0 https://github.com/NVlabs/LongLive /opt/LongLive \
#    && pip install --no-cache-dir -r /opt/LongLive/requirements.txt \
#    && pip install --no-cache-dir flash-attn --no-build-isolation
#   ENV PYTHONPATH=/opt/LongLive:$PYTHONPATH
#
# 가중치 2종류 (longlive.py 헤더 주석 참고):
#   RUN huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
#         --local-dir /opt/LongLive/wan_models/Wan2.1-T2V-1.3B
#   RUN huggingface-cli download Efficient-Large-Model/LongLive \
#         --local-dir /opt/LongLive/longlive_models
# ---------------------------------------------------------------------

COPY workers/ /app/workers/

EXPOSE 8000
ENV WM_MODEL=longlive
CMD ["python", "-m", "workers.run"]
