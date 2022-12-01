from transformers import BertModel,BertTokenizer
from torch.nn.functional import normalize
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn


class FineTunedModel:
        # prefix = "/home/zjlab/mengzhangyuan/"
        def __init__(self, device_index=0):
            self.names = set()
            self.models = {}
            self.tokenizers = {}
            self.device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

        def show_name(self)->set:
            return self.names

        def add_model(self, name, model_path, token_path=None, addprefix=False):
            if name in self.names:
                input_result = input(f"model {name} has existed, please enter [y] to replace it:\n")
                if input_result == 'y':
                    return
            self.names.add(name)
            checkpoint = FineTunedModel.prefix + model_path if addprefix else model_path
            self.models[name] = BertModel.from_pretrained(checkpoint)
            self.models[name].to(self.device)
            self.models[name].eval()
            if not token_path:
                self.tokenizers[name] = BertTokenizer.from_pretrained(checkpoint)
            else:
                self.tokenizers[name] = BertTokenizer.from_pretrained(FineTunedModel.prefix + token_path)

        # 全角字符转半角，字符串处理
        def str_q2b(self,ustring)-> str:
            new_string = ""
            for uchar in ustring:
                inside_code = ord(uchar)
                change = False
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                    change = True
                elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                    change = True
                new_string += chr(inside_code) if change else uchar
            return new_string


        def get_vectors_norm(self, data, name=None) -> torch.tensor:
            with torch.no_grad():
                data = self.get_vectors_avg(data,name)
                data = normalize(data, dim=1)
                return data

        def get_vectors_avg(self, data, name=None) -> torch.tensor:
            data = [self.str_q2b(each_data.lower()) for each_data in data]
            with torch.no_grad():
                if name:
                    tokenizer = self.tokenizers[name]
                    model = self.models[name]
                else:
                    tokenizer = self.tokenizers.values().__iter__().__next__()
                    model = self.models.values().__iter__().__next__()

                data = tokenizer(data, truncation=True, padding="longest", max_length=512, return_tensors="pt")
                data.to(self.device)
                out = model(**data, output_hidden_states=True)
                # out_data1 = out.last_hidden_state
                out_data = out.hidden_states[-1]
                attention = data["attention_mask"]
                out_data[attention == 0] = 0
                mean_data = torch.sum(out_data, axis=1) / torch.sum(attention, axis=-1).unsqueeze(-1)
                return mean_data




app = FastAPI()

class Item(BaseModel):
    strarr: List[str] = []

my_model = FineTunedModel(7)
my_model.add_model("nezha","/home/zjlab/zyg/nezha_fine_tuned")

@app.post("/str2vec/")
async def create_item(item:Item):

    str_ = item.strarr
    print(str_)
    result = my_model.get_vectors_norm(str_)
    print("=======",result)
    if my_model.device == "cpu":
        list_result = result.numpy().tolist()
    else:
        list_result = result.cpu().numpy().tolist()
    
    print("*******",list_result)

    
    return {'data': list_result}

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8789)
