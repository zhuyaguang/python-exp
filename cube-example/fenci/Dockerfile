FROM python:3.7

RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install minio
RUN python3 -m pip install bentoml

LABEL maintainer "zhuyaguang <zhuyaguang@zhejianglab.com>"

EXPOSE 3123
WORKDIR /app

COPY . /app/
RUN python hello.py
RUN bentoml serve service.py:svc --host=0.0.0.0 --port=3123 & 

# docker build -f Dockerfile -t 10.100.29.62/kubeflow/cube:v1 ./

# docker run -p 3003:3000 bilstm:u3ep7aaokslojlqy