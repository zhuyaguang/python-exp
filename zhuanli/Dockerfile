FROM python:3.7
RUN python3 -m pip install transformers
RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install tokenizers
RUN python3 -m pip install argparse
RUN python3 -m pip install datasets
RUN python3 -m pip install minio



RUN mkdir -p /home/pipeline-demo/guo-demo/bert_base_uncased_Patrick

COPY ./train.py /home/pipeline-demo/guo-demo/train.py