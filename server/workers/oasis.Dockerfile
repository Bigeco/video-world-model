# Oasis 워커. DIAMOND와 torch 버전이 다를 수 있으므로 이미지를 분리한다.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY workers/requirements-common.txt .
RUN pip install --no-cache-dir -r requirements-common.txt

# ---------------------------------------------------------------------
# Oasis 리포와 추가 의존성은 여기서 설치하세요.
#   RUN git clone https://github.com/etched-ai/open-oasis /opt/oasis \
#    && pip install --no-cache-dir -r /opt/oasis/requirements.txt
#   ENV PYTHONPATH=/opt/oasis:$PYTHONPATH
# ---------------------------------------------------------------------

COPY workers/ /app/workers/

EXPOSE 8000
CMD ["python", "-m", "workers.run"]
