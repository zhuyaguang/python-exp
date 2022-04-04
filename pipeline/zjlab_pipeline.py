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

def load_data():
    # Using readlines()
    file1 = open('trainData.txt', 'r')
    Lines = file1.readlines()
 
    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        print("Line{}: {}".format(count, line.strip()))

        
def pass_file_op():
    return dsl.ContainerOp(
        name='pass_file',
        image='zhuyaguang/pipeline:v1',
        command=['sh', '-c'],
        arguments=['cat trainData.txt'],
        file_outputs={
            'data': '/trainData.txt',
        }
    )

def train_op(text,config:str,model:str):
    return dsl.ContainerOp(
        name='trian_file',
        image='zhuyaguang/pipeline:v2',
        command = ['python', 'testdemo.py'],
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
    from transformers import BertConfig,BertTokenizer,BertForMaskedLM,DataCollatorForWholeWordMask,Trainer,TrainingArguments,LineByLineWithRefDataset
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
                            
    train_dataset = LineByLineWithRefDataset(tokenizer = tokenizer,file_path = file_path, ref_path = ref_path, block_size=512)      
            
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
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

    # print_task = print_text(url)

    load_data_op=func_to_container_op(
        func=load_data,
        base_image="zhuyaguang/pipeline:v1",  
    )
    load_data_task = load_data_op()

    pass_file_task = pass_file_op()
    train_task = train_op(pass_file_task.output,config,model)
    

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(zjlab1_pipeline, 'zjlab2.yaml')






