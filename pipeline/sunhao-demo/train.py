#!/usr/bin/env python
# coding: utf-8
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
    
)
import torch
import tokenizers
import argparse
import time
from minio import Minio
import os
import time



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    
    client = Minio('10.101.32.11:9000',access_key='admin',secret_key='root123456',secure=False)
    ## 下载 vocab.txt  word_bag.txt config.json pytorch_model.bin bert_base_uncased_Patrick
    for item in client.list_objects("data-sunhao",recursive=True):
        data = client.get_object("data-sunhao", item.object_name)
        print(item.object_name)
        with open("/home/pipeline-demo/sunhao-demo/"+item.object_name,"wb") as fp:
            for d in data.stream(1024):
                fp.write(d)
    
    # 下载数据集
    filename="claim_"+f'{args.train_time}'
    data = client.get_object("data-sunhao-claim", filename)
    with open("/home/pipeline-demo/sunhao-demo/"+filename,"wb") as fp:
        for d in data.stream(1024):
            fp.write(d)

    tokenizer_kwargs = {
        "model_max_length": 512
    }

    print("====",args.train_time)
    base_path = "/home/pipeline-demo/sunhao-demo/"+f'Epoch{args.train_time}'
    tokenizer =  BertTokenizer.from_pretrained('/home/pipeline-demo/sunhao-demo/', **tokenizer_kwargs)
    if (args.train_time == 0):
        args_config = "/home/pipeline-demo/sunhao-demo/bert_base_uncased_Patrick"
        args_model = "/home/pipeline-demo/sunhao-demo/bert_base_uncased_Patrick"
    else:
        args_config= "/home/pipeline-demo/sunhao-demo/config.json"
        args_model= "/home/pipeline-demo/sunhao-demo/pytorch_model.bin"

    args_file_path = "/home/pipeline-demo/sunhao-demo/"+filename

    config_new = BertConfig.from_pretrained(args_config)
    
    model = BertForMaskedLM.from_pretrained(args_model, config=config_new)
    
    model.resize_token_embeddings(len(tokenizer))  

    print("date start",time.asctime( time.localtime(time.time()) ))
                            
    train_dataset = LineByLineTextDataset(tokenizer = tokenizer,file_path = args_file_path, block_size=512)      

    print("date end",time.asctime( time.localtime(time.time()) ))

            
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    pretrain_batch_size=26
    num_train_epochs=5
    training_args = TrainingArguments(
        output_dir='/home/pipeline-demo/sunhao-demo/args', overwrite_output_dir=True, num_train_epochs=num_train_epochs, 
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=10000, local_rank = args.local_rank, #dataloader_pin_memory = False,
        per_device_train_batch_size=pretrain_batch_size, logging_strategy ="epoch",save_strategy = "epoch", save_total_limit = 1)
    
    print("train start",time.asctime( time.localtime(time.time()) ))

    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

    print("train start",time.asctime( time.localtime(time.time()) ))

    
    trainer.train()

    trainer.save_model(base_path)
    configPath = base_path+"/config.json"
    modelPath =  base_path+"/pytorch_model.bin"
    # 覆盖上一轮的模型
    client.fput_object("data-sunhao","config.json", configPath)
    client.fput_object("data-sunhao","pytorch_model.bin", modelPath)
    # 模型归档
    s3Path1 = f'/{args.train_time}'+"/config.json"
    s3Path2 = f'/{args.train_time}'+"/pytorch_model.bin"
    client.fput_object("result-sunhao",s3Path1, configPath)
    client.fput_object("result-sunhao",s3Path2, modelPath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nezha_train")
    #parser.add_argument("--config", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    #parser.add_argument("--model", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    #parser.add_argument("--file_path", type = str, default = "/home/pipeline-demo/sunhao-demo/claim_0", help = "二次训练_nezha")
    #parser.add_argument("--save_dir", type = str, default = "/home/pipeline-demo/sunhao-demo", help = "二次训练_nezha")
    parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
    parser.add_argument("--train_time", type = int, default = 0, help = "The num of training times")
    args = parser.parse_args()
    main(args)
