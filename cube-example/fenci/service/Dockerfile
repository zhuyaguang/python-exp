FROM python:3.8

COPY . /SVC

RUN python3 -m pip install pickle -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install FastAPI -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# docker build -f Dockerfile -t 10.100.29.62/kubeflow/svc:v1 ./