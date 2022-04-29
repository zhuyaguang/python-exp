# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection

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
# 把孙浩的向量插入milvus
#################################################################################

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="10.101.32.24", port="19530")

has = utility.has_collection("hello_milvus3")
print(f"Does collection hello_milvus3 exist in Milvus: {has}")

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus3`"))
hello_milvus = Collection("hello_milvus3", schema, consistency_level="Strong")


print(fmt.format("Start inserting entities"))

data = np.fromfile("/home/hdu-sunhao/孙浩/paper_patent_data/claims_embedding", dtype=np.float32,count=7680000)
data.shape = -1, 768
num_entities = 10000


entities2 = [
    # provide the pk field because `auto_id` is set to False
    [i for i in range(num_entities)],
    data.tolist(),  # field embeddings
]
# print(entities2)

insert_result = hello_milvus.insert(entities2)

print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

################################################################################
# 4. create index

print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
print(fmt.format("Start loading"))
hello_milvus.load()







