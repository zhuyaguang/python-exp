from transformers import BertModel,BertTokenizer
from torch.nn.functional import normalize
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import asyncio
import time


class FineTunedModel:
        # prefix = "/home/zjlab/mengzhangyuan/"
        def __init__(self, device_index=1):
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
# 创建一个最大并发请求数量为 100 的 Semaphore 对象
# semaphore = asyncio.Semaphore(100)

class Item(BaseModel):
    strarr: List[str] = []

my_model = FineTunedModel(5)
my_model.add_model("nezha","/home/zjlab/zyg/nezha_fine_tuned")

@app.post("/str2vec/")
def create_item(item:Item):
    gpu_mem_total, gpu_mem_used, gpu_mem_free,gpu_use_rate = get_gpu_mem_info(5)
    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB, 使用率 {}'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free, gpu_use_rate))

    # 在请求处理函数中使用 with 语句获取 Semaphore 对象的使用权
    #async with semaphore:
    if gpu_use_rate <= 0.9:
        str_ = item.strarr
        print('本次请求长度',len(str_))
        hash_value = ''
        if len(str_)>0:
            hash_value = hash(str_[0])
            print('本次请求第一个元素的哈希和值：', hash_value, str_[0])
        start_time = time.time()
        result = my_model.get_vectors_norm(str_)
        if my_model.device == "cpu":
            list_result = result.numpy().tolist()
            end_time = time.time()
            print(f"向量化(cpu)执行时间为：{elapsed_time}秒, 哈希值为{hash_value}")
        else:
            list_result = result.cpu().numpy().tolist()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"向量化(gpu)执行时间为：{elapsed_time}秒, 哈希值为{hash_value}")
        
        print("=======结果向量的长度",len(list_result), "哈希值为", hash_value)

        end_time = time.time()

        print(f"向量化(return前)执行时间为：{elapsed_time}秒, 哈希值为{hash_value}")
        return {'data': list_result}
    else:
        raise HTTPException(status_code=419, detail="显存不足，请放低请求频率")
    
def get_gpu_mem_info(gpu_id=1):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    start_time = time.time()
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"获取显卡号码执行时间为：{elapsed_time}秒")

    start_time = time.time()
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    userate = used/total
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"获取显存使用情况执行时间为：{elapsed_time}秒")
    return total, used, free,userate


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8791)
