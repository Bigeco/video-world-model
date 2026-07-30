# DIAMOND(Atari) 워커. `main` 브랜치 코드. CS:GO는 별도 브랜치/이미지(diamond-csgo.Dockerfile).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY workers/requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# ---------------------------------------------------------------------
# DIAMOND 리포 (main 브랜치 = Atari):
#   RUN git clone https://github.com/eloialonso/diamond /opt/diamond \
#    && pip install --no-cache-dir -r /opt/diamond/requirements.txt \
#    && pip install --no-cache-dir "autorom[accept-rom-license]" && AutoROM -y
#   ENV PYTHONPATH=/opt/diamond/src:$PYTHONPATH
# 저장소 코드가 src/ 를 임포트 루트로 쓰므로 PYTHONPATH는 저장소 루트가 아니라
# 반드시 <저장소>/src 여야 한다 (diamond_atari.py 헤더 주석 참고).
# ---------------------------------------------------------------------

COPY workers/ /app/workers/

EXPOSE 8000
ENV WM_MODEL=diamond-atari
CMD ["python", "-m", "workers.run"]
