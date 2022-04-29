# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.


import random

from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)


def hello_milvus():
    # create connection
    connections.connect(host='10.101.4.13',port='19530')

    print(f"\nList collections...")
    print(list_collections())



    print(f"\nCreate collection...")
    collection = Collection(name="gosdk_index_example", schema=default_schema)

    print(f"\nList collections...")
    print(list_collections())




    print(f"\nload collection...")
    collection.load()

    # load and search
    topK = 5
    round_decimal = 3
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    import time
    start_time = time.time()
    print(f"\nSearch...")
    # define output_fields of search result
    res = collection.search(
        vectors[-2:], "float_vector", search_params, topK, "count > 100",
        output_fields=["count", "random_value"], round_decimal=round_decimal
    )
    end_time = time.time()

    # show result
    for hits in res:
        for hit in hits:
            # Get value of the random value field for search result
            print(hit, hit.entity.get("random_value"))
    print("search latency = %.4fs" % (end_time - start_time))

    #drop collection
    collection.drop()


hello_milvus()