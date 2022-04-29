#!/usr/bin/env python3
# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp.dsl as dsl
import kfp.components as components
from typing import NamedTuple
import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath
from kubernetes.client.models import V1ContainerPort

# 方法1 baseimage+command
# 方法2 func_to_container_op （demo1.py） 需要拉取 gcr.io/google-containers/busybox         
def pass_file_op():
    return dsl.ContainerOp(
        name='pass_file',
        image='zhuyaguang/pipeline:v4',
        command=['sh', '-c'],
        arguments=['cat /pipelines/component/src/trainData.txt '],
        file_outputs={
            'data': '/pipelines/component/src/trainData.txt',
        }
    )

# 拷贝文件到根目录
def copy_op():
    return dsl.ContainerOp(
        name='copy_file',
        image='zhuyaguang/pipeline:v4',
        command=['sh', '-c'],
        arguments=['cp /pipelines/component/src/trainData.txt .'],
        file_outputs={
            'data': '/trainData.txt',
        }
    )


# base镜像+command
def train_op(text,config:str,model:str):
    return dsl.ContainerOp(
        name='trian_file',
        image='zhuyaguang/pipeline:v4',
        command = ['python3', '/pipelines/component/src/testdemo.py'],
        arguments=[
            "--config", config,
            "--model", model
            ],
    )

# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
    '''Print text'''
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="nezha_train")
#     parser.add_argument("--config", type = str, default = None, help = "二次训练_nezha")
#     parser.add_argument("--model", type = str, default = None, help = "二次训练_nezha")
#     parser.add_argument("--file_path", type = str, default = None, help = "二次训练_nezha")
#     parser.add_argument("--ref_path", type = str, default = None, help = "二次训练_nezha")
#     parser.add_argument("--save_dir", type = str, default = None, help = "二次训练_nezha")
#     parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
#     args = parser.parse_args()
#     train(args)
def train(config:str,model:str,file_path:str,ref_path:str,save_dir:str,local_rank:int):
    from transformers import BertConfig,BertTokenizer,BertForMaskedLM,DataCollatorForLanguageModeling,Trainer,LineByLineTextDataset,TrainingArguments
    import torch
    import tokenizers
    import argparse

    tokenizer_kwargs = {
        "model_max_length": 512
    }
    
    tokenizer =  BertTokenizer.from_pretrained('/home/hdu-sunhao/孙浩/二次训练_nezha/', **tokenizer_kwargs)
    
    config_new = BertConfig.from_pretrained(config)
    
    model = BertForMaskedLM.from_pretrained(model, config=config_new)
    
    model.resize_token_embeddings(len(tokenizer))  
                            
    train_dataset = LineByLineTextDataset(tokenizer = tokenizer,file_path = file_path, ref_path = ref_path, block_size=512)      
            
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    pretrain_batch_size=16
    num_train_epochs=5
    training_args = TrainingArguments(
        output_dir='/home/hdu-sunhao/孙浩/二次训练_nezha/model-claims/args', overwrite_output_dir=True, num_train_epochs=num_train_epochs, 
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=10000, local_rank = local_rank, #dataloader_pin_memory = False,
        per_device_train_batch_size=pretrain_batch_size, logging_strategy ="epoch",save_strategy = "epoch", save_total_limit = 1)
    
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)
    
    trainer.train()
    trainer.save_model(save_dir)
@dsl.pipeline(
    name='zjlab-pipeline',
    description='A pipeline of zjlab to train data.'
)
def zjlab1_pipeline(url='gs://ml-pipeline/sample-data/shakespeare/shakespeare1.txt',config="default config",model="default model"):
    """A pipeline load data to train."""


    pass_file_task = pass_file_op()

    print_task = print_text(pass_file_task.output)
    # 运行一个带参数的python函数

    # 方法1 base镜像+command
    copy_task = copy_op()
   
    train_task = train_op(copy_task.output,config,model)
    

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(zjlab1_pipeline, 'zjlab3.yaml')






