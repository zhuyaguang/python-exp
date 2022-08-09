#!/usr/bin/env python
# coding: utf-8
from minio import Minio
import os


def main():
   
    client = Minio('10.101.32.11:9000',access_key='admin',secret_key='root123456',secure=False)

    data = client.get_object("data", "bilstm_crf-jbgihzqh62lojlqy.bentomodel")
    with open("./data/bilstm_crf-jbgihzqh62lojlqy.bentomodel","wb") as fp:
        for d in data.stream(1024):
            fp.write(d)

    os.system('bentoml models import ./data/bilstm_crf-jbgihzqh62lojlqy.bentomodel')

    data = client.get_object("data", "BiLSTM_CRF_voc.txt")
    with open("./data/BiLSTM_CRF_voc.txt","wb") as fp:
        for d in data.stream(1024):
            fp.write(d)
    
    data = client.get_object("data", "BiLSTM_CRF_tags.txt")
    with open("./data/BiLSTM_CRF_tags.txt","wb") as fp:
        for d in data.stream(1024):
            fp.write(d)


if __name__ == "__main__":
    main()