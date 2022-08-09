FROM python:3.7

RUN python3 -m pip install minio

LABEL maintainer "zhuyaguang <zhuyaguang@zhejianglab.com>"

COPY ./download-data.py .


# docker build -f Dockerfile -t 10.100.29.62/kubeflow/cube:v1 ./
