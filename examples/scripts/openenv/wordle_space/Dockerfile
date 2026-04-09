FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install
ENV GIT_CLONE_PROTECTION_ACTIVE=false
RUN pip install --no-build-isolation "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle" \
    || (git clone https://huggingface.co/spaces/openenv/wordle /tmp/wordle && pip install /tmp/wordle && rm -rf /tmp/wordle)

COPY app.py /app/app.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

WORKDIR /app
