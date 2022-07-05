#!/usr/bin/env python
# coding: utf-8
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
import tokenizers
import argparse
import time
from minio import Minio
import os
from datasets import load_dataset
import zipfile

def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))


def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    client = Minio('10.101.32.11:9000',access_key='admin',secret_key='root123456',secure=False)
    ## 下载 vocab.txt  word_bag.txt config.json pytorch_model.bin bert_base_uncased_Patrick
    ## mkdir bert_base_uncased_Patrick first 
    for item in client.list_objects("data-sunhao",recursive=True):
        data = client.get_object("data-sunhao", item.object_name)
        print(item.object_name)
        with open("/home/pipeline-demo/guo-demo/"+item.object_name,"wb") as fp:
            for d in data.stream(1024):
                fp.write(d)
    
    # 下载数据集
    filename="patent_abstract_"+f'{args.train_time}'+".csv"
    data = client.get_object("data-abstract", filename)
    with open("/home/pipeline-demo/guo-demo/"+filename,"wb") as fp:
        for d in data.stream(1024):
            fp.write(d)

    max_length= 512

    print("====",args.train_time)
    base_path = "/home/pipeline-demo/guo-demo/"+f'Epoch{args.train_time}'
    tokenizer =  BertTokenizer.from_pretrained('/home/pipeline-demo/guo-demo/', max_length=max_length)
    if (args.train_time == 1):
        args_config = "/home/pipeline-demo/guo-demo/bert_base_chinese_Patrick_von_Platen_"
        args_model = "/home/pipeline-demo/guo-demo/bert_base_chinese_Patrick_von_Platen_"
    else:
        args_config= "/home/pipeline-demo/guo-demo/config.json"
        args_model= "/home/pipeline-demo/guo-demo/pytorch_model.bin"

    args_file_path = "/home/pipeline-demo/guo-demo/"+filename

    config_new = BertConfig.from_pretrained(args_config)
    
    model = BertForMaskedLM.from_pretrained(args_model, config=config_new)
    
    model.resize_token_embeddings(len(tokenizer))  

    print("date start",time.asctime( time.localtime(time.time()) ))
                            

    raw_datasets = load_dataset('csv', data_files= args_file_path, delimiter='\t', column_names=['id', 'publication_no', 'abstract'],  split='train')


    def tokenize_function(examples, text_column_name="abstract"):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if line and len(line) > 0 and not line.isspace()
        ]
        
        return tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
    tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=50,
                remove_columns=["id", "publication_no"],
                desc="Running tokenizer on dataset line_by_line",
            )

    dataset = tokenized_datasets.train_test_split(test_size=0.1, shuffle=True, seed=2022)
    # print(dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]      

    print("date end",time.asctime( time.localtime(time.time()) ))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # print(data_collator.__class__)
    # exit()
    
    pretrain_batch_size=26
    num_train_epochs=5
    training_args = TrainingArguments(
        output_dir=base_path, overwrite_output_dir=True, num_train_epochs=num_train_epochs, 
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=10000, local_rank = args.local_rank, 
        per_device_train_batch_size=pretrain_batch_size,logging_steps=500, save_total_limit = 1, #logging_dir="./runs",
        load_best_model_at_end=True, save_strategy="epoch", evaluation_strategy="epoch",
        metric_for_best_model="loss")
    
    my_callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)
                    # TensorBoardCallback(tb_writer=None)

    ]
    trainer = Trainer(
        model=model, args=training_args, 
        data_collator=data_collator, 
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        callbacks=my_callbacks
        )


    print("train start",time.asctime( time.localtime(time.time()) ))

    
    trainer.train()

    trainer.save_model(base_path)
    configPath = base_path+"/config.json"
    modelPath =  base_path+"/pytorch_model.bin"

    # 覆盖上一轮的模型
    client.fput_object("data-sunhao","config.json", configPath)
    client.fput_object("data-sunhao","pytorch_model.bin", modelPath)
    # 模型归档 先压缩再上传
    input_path = base_path
    output_path = "./"+f'Epoch{args.train_time}'+".zip"
 
    zipDir(input_path, output_path)
    client.fput_object("result-guo",f'Epoch{args.train_time}'+".zip", output_path)
    # 日志归档





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nezha_train")
    #parser.add_argument("--config", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    #parser.add_argument("--model", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    #parser.add_argument("--file_path", type = str, default = "/home/pipeline-demo/sunhao-demo/claim_0", help = "二次训练_nezha")
    #parser.add_argument("--save_dir", type = str, default = "/home/pipeline-demo/sunhao-demo", help = "二次训练_nezha")
    parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
    parser.add_argument("--train_time", type = int, default = 1, help = "The num of training times")
    args = parser.parse_args()
    main(args)

