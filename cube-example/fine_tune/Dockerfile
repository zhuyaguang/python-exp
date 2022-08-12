#FROM nvcr.io/nvidia/pytorch:21.09-py3

FROM python:3.7

RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install transformers
RUN python3 -m pip install datasets
RUN python3 -m pip install argparse


LABEL maintainer "zhuyaguang <zhuyaguang@zhejianglab.com>"

COPY ./fine_tune.py .


# docker build -f Dockerfile -t 10.100.29.62/kubeflow/trainM:v1 ./

# python fine_tune.py  --recall_path=name --target=abstract --batch_size_train=12 --batch_size_eval=256 --gpu=0 --data_index=0 --model_path=/mnt/admin/pre-train/nezha --data_path="/mnt/admin/pre-train"
