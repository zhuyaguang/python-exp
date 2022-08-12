import argparse
from minio import Minio
import os.path
from os import path

def main(args):

    print("第一个参数是",args.file_path)
    print("第二个参数是",args.bucket_name)
    print("第三个参数是",args.output)
    print("第四个参数是",args.minio_endpoint)
    print("第五个参数是",args.access_key)
    print("第六个参数是",args.secret_key)

    client = Minio(args.minio_endpoint,access_key=args.access_key,secret_key=args.secret_key,secure=False)

    # data = client.get_object("data", "bilstm_crf-jbgihzqh62lojlqy.bentomodel")
    # with open("./data/bilstm_crf-jbgihzqh62lojlqy.bentomodel","wb") as fp:
    #     for d in data.stream(1024):
    #         fp.write(d)
    
    for item in client.list_objects(args.bucket_name,recursive=True):
        data = client.get_object(args.bucket_name, item.object_name)
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
    parser.add_argument("--bucket_name", type = str, default = "test", help = "input minio bucket name")
    parser.add_argument("--output", type = str, default = "/mnt/admin/", help = "output save path")
    parser.add_argument("--minio_endpoint", type = str, default = "10.101.32.11:9000", help = "minio endpoint")
    parser.add_argument("--access_key", type = str, default = "admin", help = "minio user name")
    parser.add_argument("--secret_key", type = str, default = "root123456", help = "minio password")
    args = parser.parse_args()
    main(args)



