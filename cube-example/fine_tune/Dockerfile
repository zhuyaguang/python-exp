FROM python:3.7

RUN python3 -m pip install transformers
RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install tokenizers
RUN python3 -m pip install argparse

RUN python3 -m pip install minio

LABEL maintainer "zhuyaguang <zhuyaguang@zhejianglab.com>"

COPY ./fine_tune.py .


# docker build -f Dockerfile -t 10.100.29.62/kubeflow/cube:v1 ./
