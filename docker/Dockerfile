FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace/src

CMD ["/bin/bash"]
