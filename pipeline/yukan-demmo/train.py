import pandas as pd
import numpy as np
import os
import json
import time
import random
import torch
import torch.nn as nn
from tqdm.notebook import tqdm_notebook
from transformers import AdamW, BertTokenizer, BertForMaskedLM, BertModel
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from minio import Minio

client = Minio('10.101.32.11:9000',access_key='admin',secret_key='root123456',secure=False)


for item in client.list_objects("data",recursive=True):
    data = client.get_object("data", item.object_name)
    print(item.object_name)
    with open("/home/pipeline-demo/yukan-demo/"+item.object_name,"wb") as fp:
        for d in data.stream(1024):
            fp.write(d)

ask_and_answer_path = '/home/pipeline-demo/yukan-demo/ask_and_answer.json'
download_model_path = '/home/pipeline-demo/yukan-demo/result/'

batch_size = 32
epochs = 2
max_length = 64
seed = 900

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


tokenizer = BertTokenizer.from_pretrained('/home/pipeline-demo/yukan-demo/bert-base-chinese/')
model = BertForMaskedLM.from_pretrained('/home/pipeline-demo/yukan-demo/bert-base-chinese/')
# model = BertModel.from_pretrained('bert-base-chinese')

class LineByLineTextDataset(Dataset):  
    def __init__(self):
        data_text = LineByLineTextDataset._read_ask_and_answer()
        # print(data)
        self.data_token = tokenizer(
          text=data_text,
          padding='max_length',
          max_length=max_length,
          truncation=True, #超过max_length长度的自动截断
          return_tensors='pt'
        )['input_ids']
        # print(self.data_token)
          
    def __len__(self):
        return len(self.data_token)
  
    def __getitem__(self, idx):
        return self.data_token[idx]
  
    @staticmethod
    def _read_ask_and_answer():
        # json解析ask_and_answer文件
        with open(ask_and_answer_path, 'r') as ask_and_answer:
            records = json.load(ask_and_answer)['RECORDS']
        # 读取ask_and_answer中的问题
        data = list()
        for record in records:
            data.append(LineByLineTextDataset._drop_questionmark(record['title']))
            for similar_ask in json.loads(record['similar_ask']):
                if similar_ask['ask'] == '': continue
                data.append(LineByLineTextDataset._drop_questionmark(similar_ask['ask']))
        return data
  
    @staticmethod
    def _drop_questionmark(text):
        return text[:-1] if text[-1] == "?" or text[-1] == "？" else text
    
dataset = LineByLineTextDataset()
print("样本数量==>  input_ids  ", dataset.__len__())

dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

class Trainer:
    def __init__(self, model, dataloader, tokenizer, mlm_probability=0.15, lr=1e-4, with_cuda=True, cuda_devices=None, log_freq=10):
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.is_parallel = False                        # 多GPU 数据并行？
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability                 # masked的概率
        self.log_freq = log_freq

        # 多GPU训练
    #     if with_cuda and torch.cuda.device_count() > 1:
    #         print(f"Using {torch.cuda.device_count()} GPUS for BERT")
    #         self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
    #         self.is_parallel = True
        self.model.train()
        self.model.to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=1e-4)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
      
    def train(self, epochs):
        for epoch in range(epochs):
            ave_loss = self.iteration(epoch, self.dataloader)
            #if (epoch + 1) % 5 == 0:
            if True :
                #model_path = download_model_path + f'Epoch{epoch + 1}/.'
                base_path= f'Epoch{epoch + 1}_Batchsize{batch_size}_Loss{ave_loss:f}_DateTime{time.strftime("%Y%m%d_%H%M%S", time.localtime())}/'
                second_path= download_model_path + base_path
                model_path = second_path +"."
                print(f"{ave_loss:f}")
                print(model_path)
                model.save_pretrained(model_path)
                print(f'Download into {model_path}')
                configPath = second_path+"config.json"
                modelPath =  second_path+"pytorch_model.bin"
                s3Path1 = base_path+"config.json"
                s3Path2 = base_path+"pytorch_model.bin"
                print("==========")
                print(configPath,modelPath,s3Path1,s3Path2)
                client.fput_object("result",s3Path1, configPath)
                client.fput_object("result",s3Path2, modelPath)

      
    def iteration(self, epoch, dataloader, train=True):
        str_code = 'Train'
        total_loss = 0.0
        #with tqdm_notebook(total=len(dataloader), desc='Epoch %d Training' %epoch, ncols = 800) as pbar:
        for i,batch in enumerate(dataloader):
            # print(batch)
            print(i)
            print(len(dataloader))
            inputs, labels = self._mask_tokens(batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(input_ids=inputs, labels=labels)
            loss = outputs.loss.mean()

            if train:
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

            total_loss += loss.item()
            ave_loss = total_loss/(i+1)          
                # pbar.set_postfix(loss=float(ave_loss))
                # pbar.update(1)
        return ave_loss        
          
    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Masked Language Model """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # 使用mlm_probability填充张量
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 获取special token掩码
        # Returns:  `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        special_tokens_mask = [
              self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 将special token位置的概率填充为0
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
          # padding掩码
          padding_mask = labels.eq(tokenizer.pad_token_id)
          # 将padding位置的概率填充为0
          probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 对token进行mask采样
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 #loss只计算masked

        # 80%的概率将masked token替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的概率将masked token替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 余下的10%不做改变
        return inputs, labels

trainer = Trainer(model, dataloader, tokenizer)

trainer.train(epochs)
for i in range(10): torch.cuda.empty_cache()