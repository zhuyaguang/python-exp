import uvicorn
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer
)
import numpy as np
import random
import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

#################################################################################
# 起一个API来进行向量检索，把检索结果放在二维数组里面返回
#################################################################################
app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None





@app.post("/items/")
async def create_item(item:Item):
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="10.101.32.24", port="19530")

    has = utility.has_collection("hello_milvus3")
    print(f"Does collection hello_milvus3 exist in Milvus: {has}")
   
    tokenizer_kwargs = {
        "model_max_length": 512
    }
    tokenizer = BertTokenizer.from_pretrained('/home/hdu-sunhao/孙浩/二次训练_nezha/', **tokenizer_kwargs)
    config_new = BertConfig.from_pretrained('/home/hdu-sunhao/孙浩/二次训练_nezha/model-claims/model/model_7')
    model = BertModel.from_pretrained("/home/hdu-sunhao/孙浩/二次训练_nezha/model-claims/model/model_7", config = config_new)

    str_ = item.name
    print(str_)

    encoded_input = tokenizer(str_, truncation = True, padding = True, return_tensors='pt')
    output = model(**encoded_input)
    output = output.last_hidden_state.detach()
    print(output.shape)
    output = output.cpu()
    output = output.numpy().squeeze(axis=0)

    result = np.mean(output, axis=0)

    print(fmt.format("Start searching based on vector similarity"))
    
    collection = Collection("hello_milvus3")      # Get an existing collection.
    collection.load()
    vectors_to_search = [
        result.tolist()
    ]
    search_params = {
        "metric_type": "l2",
        "params": {"nprobe": 10},
    }
    result2 = collection.search(vectors_to_search, "embeddings", search_params, limit=3, expr=None)

    data_result=[[]]
    for hits in result2:
        for hit in hits:
            print(f"hit: {hit.entity.id}")
            data = pd.read_csv("/home/hdu-sunhao/孙浩/paper_patent_data/claims_notna.csv", skiprows=hit.entity.id+1,nrows=0,usecols = [0,1])
            data_result.append(data)
            print(data)
    

    #data2=pd.read_csv("/home/hdu-sunhao/孙浩/paper_patent_data/",nrows=hit.entity.id+1)
    #print(data2)



    collection.release()




    return data_result

# uvicorn main:app --host '0.0.0.0' --port 8123 --reload
if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8123)



