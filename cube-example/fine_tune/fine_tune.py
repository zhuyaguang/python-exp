import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split
import transformers
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    BertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import argparse
import time
import os
import numpy as np
import random
from datasets import load_dataset

def set_seed(seedval):
    random.seed(seedval)
    np.random.seed(seedval)
    torch.manual_seed(seedval)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class TripletLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #         print('start')
        # forward pass
#         anchor_data, positive_data, negative_data = inputs.values()
        lis = ['input_ids', 'token_type_ids', 'attention_mask']
        data = [{eachname:inputs[eachname+str(index) if index else eachname]for eachname in lis} for index in range(inputs["size"])]
#         anchor_data = {each:inputs[each+'0'] for each in lis}
#         positive_data = anchor_data
#         negative_data = anchor_data
#         anchor_output = self.use_avg_2(anchor_data, model)
#         positive_output = self.use_avg_2(positive_data, model)
#         negative_output = self.use_avg_2(negative_data, model)
        data = [self.use_avg_2(each,model) for each in data]
        triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                        margin=0.9)
        # compute custom loss
        loss = triplet_loss(*data)
        #         print('finish')
        return (loss,data) if return_outputs else loss

    def use_avg_2(self, input_data, model):
        out = model(**input_data, output_hidden_states=True)
        #         outdata1 = out.last_hidden_state
        outdata = out.hidden_states[-1]
        attention = input_data["attention_mask"]
        outdata[attention == 0] = 0
        # outdata = outdata[:, 1:-1, :]
        #             print(outdata.size())
        meandata = torch.sum(outdata, axis=1) / torch.sum(attention, axis=-1).unsqueeze(-1)
        return meandata


import pandas as pd  # 这个包用来读取CSV数据


class mydataset(Dataset):
    def __init__(self, file_path):  # self参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        self.csv_data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return self.csv_data.values[idx]


# !/usr/bin/env python
# coding: utf-8
def train(args):
    #     transformers.logging.set_verbosity_info()
    #   transformers.logging.set_verbosity_warning()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    max_length = 512
    set_seed(1994)
    path_for_token = f"{args.data_path}/nezha/"
    base_path = f'/mnt/admin/pre-train/Epoch-triplet-{args.recall_path}-{args.target}'
    tokenizer = BertTokenizer.from_pretrained(path_for_token, max_length=max_length)
    # args_file_path = f"./triplet_loss_data/{args.recall_path}_{args.target}_data_{args.data_index}.csv" if args.data_index else f"./triplet_loss_data/{args.recall_path}_{args.target}_data.csv"
    args_file_path = f"{args.data_path}/triplet_loss_data/{args.recall_path}_{args.target}_data_{args.data_index}.csv"
    print(f"========Now training for {args.recall_path}-{args.target} with data index {args.data_index}")
    print(f"Training on gpu:{args.gpu}")
    print(f"Load model from checkpoint:{args.model_path}")
    model = BertModel.from_pretrained(args.model_path)
    #raw_datasets = load_dataset('csv', data_files=args_file_path,
    #                           column_names=['name1', 'name2', 'name3'], split='train')
    train_batch_size = args.batch_size_train
    eval_batch_size = args.batch_size_eval
    num_train_epochs = 3
    def get_token(lis):
        return tokenizer(
            lis,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt')
    lis = ['input_ids', 'token_type_ids', 'attention_mask']
    
    def collator(input,size=3 ):
        #         print(type(input))
        #         print(input)
        #         print(len(input))
        
        data = []
        for index in range(size):
            data.append([each[index] for each in input])   
        data = [get_token(each) for each in data]
        result = {}
        for index,val in enumerate(data):
            for each in lis:
                result[each+str(index) if index else each]  = val[each]
        result["size"]  = size
        return result
        

    # #     training_args = TrainingArguments(
    # #         output_dir=base_path, overwrite_output_dir=True, num_train_epochs=num_train_epochs,
    # #         learning_rate=1e-4, weight_decay=0.01, warmup_steps=10000, local_rank=args.local_rank,
    # #         per_device_train_batch_size=pretrain_batch_size,  save_total_limit=1,
    # #         save_strategy="epoch", logging_strategy="epoch",
    # #         metric_for_best_model="loss")
    training_args = TrainingArguments(
        output_dir=base_path, overwrite_output_dir=True, num_train_epochs=num_train_epochs,
        learning_rate=5e-5, weight_decay=0.01, warmup_steps=10000, local_rank=args.local_rank,
        per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,save_total_limit=5,
        save_strategy="steps", evaluation_strategy="steps",logging_steps=4000,save_steps=4000,eval_steps=4000,
        load_best_model_at_end  =True,
        metric_for_best_model="eval_loss", logging_dir="temp",seed=1994,data_seed=1994)
    # print(training_args.greater_is_better)
    # #     # my_callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)
    # #     #                 # TensorBoardCallback(tb_writer=None)
    # #     #
    # #     #                 ]
    # #     trainer = Trainer(
    # #         model=model, args=training_args,
    # #         data_collator=data_collator,
    # #         train_dataset=train_dataset, eval_dataset=eval_dataset,
    # #     )

    custom_dataset = mydataset(args_file_path)
    eval_ratio = 0.005
    train_size = int(len(custom_dataset)*(1-eval_ratio))
    eval_size = len(custom_dataset)-train_size
    print(f"the evaluation size is {eval_size}")
    train_dataset, eval_dataset = random_split(custom_dataset,[train_size,eval_size],generator=torch.Generator().manual_seed(1994))
    trainer = TripletLossTrainer(
        model = model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    transformers.utils.logging.set_verbosity_info()
    trainer.train()
    trainer.save_model(base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triplet_loss_fine_tune")
    parser.add_argument("--recall_path",type=str,required=True,help="the recall path(name,abstract,claim)")
    parser.add_argument("--target",type=str,required=True,help="domain(name,abstract,claim) to fine tune")
    parser.add_argument("--batch_size_train",type=int,required=True,help="train batch size")
    parser.add_argument("--batch_size_eval",type=int,required=True,help="eval batch size")
    parser.add_argument("--model_path",type=str,help="/mnt/admin/pre-train/nezha")
    parser.add_argument("--gpu",type=str,required=True,help="GPU to use")
    parser.add_argument("--data_index",type=str,help="set the piece of data")
    parser.add_argument("--data_path",type=str,help="/mnt/admin/pre-train")
    args = parser.parse_args()
    args.train_time = 1
    args.local_rank = -1
    train(args)
