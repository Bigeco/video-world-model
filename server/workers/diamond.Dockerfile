# DIAMOND 워커 (CS:GO + Atari).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY workers/requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# ---------------------------------------------------------------------
# DIAMOND 리포:
#   RUN git clone https://github.com/eloialonso/diamond /opt/diamond \
#    && pip install --no-cache-dir -r /opt/diamond/requirements.txt
#   ENV PYTHONPATH=/opt/diamond/src:$PYTHONPATH
# Atari 변종을 쓰려면 ALE ROM도 필요합니다:
#   RUN pip install --no-cache-dir "autorom[accept-rom-license]" && AutoROM -y
# ---------------------------------------------------------------------

COPY workers/ /app/workers/

EXPOSE 8000
CMD ["python", "-m", "workers.run"]
