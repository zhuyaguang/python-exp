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

def main(args):

    tokenizer_kwargs = {
        "model_max_length": 512
    }
    
    tokenizer =  BertTokenizer.from_pretrained('/home/pipeline-demo/', **tokenizer_kwargs)
    
    config_new = BertConfig.from_pretrained(args.config)
    
    model = BertForMaskedLM.from_pretrained(args.model, config=config_new)
    
    model.resize_token_embeddings(len(tokenizer))  
                            
    train_dataset = LineByLineTextDataset(tokenizer = tokenizer,file_path = args.file_path, block_size=512)      
            
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    pretrain_batch_size=16
    num_train_epochs=5
    training_args = TrainingArguments(
        output_dir='/home/pipline-demo/args', overwrite_output_dir=True, num_train_epochs=num_train_epochs, 
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=10000, local_rank = args.local_rank, #dataloader_pin_memory = False,
        per_device_train_batch_size=pretrain_batch_size, logging_strategy ="epoch",save_strategy = "epoch", save_total_limit = 1)
    
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)
    
    trainer.train()
    trainer.save_model(args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nezha_train")
    parser.add_argument("--config", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    parser.add_argument("--model", type = str, default = "bert-base-uncased", help = "二次训练_nezha")
    parser.add_argument("--file_path", type = str, default = "/home/pipeline-demo/newfileaa", help = "二次训练_nezha")
    parser.add_argument("--save_dir", type = str, default = "/home/pipeline-demo", help = "二次训练_nezha")
    parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
    args = parser.parse_args()
    main(args)
