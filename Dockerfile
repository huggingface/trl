FROM python:3.7-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

WORKDIR $APP_HOME

RUN apt update && \
    apt install -y watch vim git curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y --no-install-recommends git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# needs to be compiled for latest cuda to work on high end GPUs
RUN pip3 install --no-cache-dir torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install protobuf~=3.19.0
RUN pip3 install ipython
RUN pip3 install pdbpp
RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

COPY . /app
RUN pip3 install -r /app/requirements.txt

CMD ["sleep", "infinity"]
