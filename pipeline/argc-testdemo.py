import argparse

def main(args):

    print("第一个参数是",args.config)
    print("第二个参数是",args.model)

    # Using readlines()
    file1 = open('trainData.txt', 'r')
    Lines = file1.readlines()
 
    count = 0
    # Strips the newline character
    print("模型文件数据")
    for line in Lines:
        count += 1
        print("Line{}: {}".format(count, line.strip()))

    # Program to show various ways to read and
    # write data in a file.
    file1 = open("trainData.txt","w")
    L = [args.config," ",args.model," London \n"] 
  
    # \n is placed to indicate EOL (End of Line)
    file1.write("Hello \n")
    file1.writelines(L)
    file1.close() #to change file access modes
  
    file1 = open("trainData.txt","r+") 
  
    print("Output of Read function is ")
    print(file1.read())
    print()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nezha_train")
    parser.add_argument("--config", type = str, default = None, help = "二次训练_nezha")
    parser.add_argument("--model", type = str, default = None, help = "二次训练_nezha")
    args = parser.parse_args()
    main(args)