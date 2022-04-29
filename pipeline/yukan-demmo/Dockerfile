FROM python:3.7
RUN python3 -m pip install transformers
RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install pandas
RUN python3 -m pip install numpy

RUN mkdir -p /home/pipeline-demo/yukan-demo/result
COPY ./ask_and_answer.json /home/pipeline-demo/yukan-demo/ask_and_answer.json
COPY ./bert-base-chinese /home/pipeline-demo/yukan-demo/bert-base-chinese
COPY ./train.py /home/pipeline-demo/yukan-demo/train.py

