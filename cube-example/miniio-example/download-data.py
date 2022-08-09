import argparse
from minio import Minio
import os.path
from os import path

def main(args):

    print("第一个参数是",args.file_path)
    print("第二个参数是",args.dic_path)
    print("第三个参数是",args.output)

    client = Minio('10.101.32.11:9000',access_key='admin',secret_key='root123456',secure=False)

    # data = client.get_object("data", "bilstm_crf-jbgihzqh62lojlqy.bentomodel")
    # with open("./data/bilstm_crf-jbgihzqh62lojlqy.bentomodel","wb") as fp:
    #     for d in data.stream(1024):
    #         fp.write(d)
    
    for item in client.list_objects(args.dic_path,recursive=True):
        data = client.get_object(args.dic_path, item.object_name)
        print(item.object_name)
    
        dict = os.path.dirname(args.output+item.object_name)
        exist = path.exists(dict)
        print(dict,exist)
        if  exist == False:
            os.makedirs(dict)
            

        with open(args.output+item.object_name,"wb") as fp:
            for d in data.stream(1024):
                fp.write(d)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download train data from minio")
    parser.add_argument("--file_path", type = str, default = None, help = "input minio file path")
    parser.add_argument("--dic_path", type = str, default = None, help = "input minio dictionary path")
    parser.add_argument("--output", type = str, default = "/mnt/admin/", help = "output save path")
    args = parser.parse_args()
    main(args)



