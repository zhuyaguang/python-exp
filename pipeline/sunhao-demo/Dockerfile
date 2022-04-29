FROM python:3.7
RUN python3 -m pip install transformers
RUN python3 -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install tokenizers
RUN python3 -m pip install argparse
COPY ./vocab.txt /home/pipeline-demo/vocab.txt
COPY ./newfileaa /home/pipeline-demo/newfileaa
COPY ./preData.py /home/pipeline-demo/train.py