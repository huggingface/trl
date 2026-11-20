# TermGrade's base image (ai-and/termgrade-environments base_image/Dockerfile, verbatim) plus the tmux + asciinema step
# every TermGrade task Dockerfile adds for terminus-2. Built as a Docker Space so HF Sandboxes can pull it as
# hf.co/spaces/<owner>/<space>; the per-task part of each TermGrade Dockerfile (its setup.sh) runs at sandbox start
# instead, from the task's healthcheck (see prepare.py).
FROM python:3.12.13-bookworm

RUN pip install --no-cache-dir pytest numpy

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gzip coreutils findutils grep sed gawk file jq curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_INSTALL_DIR=/usr/local/bin
ENV UV_CACHE_DIR=/opt/uv/cache

RUN set -eux; \
    (command -v uv >/dev/null 2>&1 || ( \
        apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates \
     && curl -LsSf https://astral.sh/uv/0.10.11/install.sh | sh )); \
    uv venv /opt/verifier_venv --python /usr/local/bin/python3 --system-site-packages; \
    uv pip install --python /opt/verifier_venv pytest==8.4.1 pytest-json-ctrf==0.3.5; \
    /opt/verifier_venv/bin/pytest --version; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

# From every TermGrade task Dockerfile: terminus-2 needs tmux + asciinema.
RUN set +e; \
    command -v tmux >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends tmux) || apk add --no-cache tmux; \
    command -v asciinema >/dev/null 2>&1 || pip install --no-cache-dir --quiet asciinema || pip3 install --no-cache-dir --quiet asciinema; \
    rm -rf /var/lib/apt/lists/*; true
RUN mkdir -p /solution /tests

# Only so the Space itself has something to serve; HF Sandboxes start their own process.
CMD ["python", "-m", "http.server", "7860"]
