from filesplit.split import Split
import os
base = "/home/jl/trace_data/"
trace_base = base + "raw_traces/"
def mkdir(path):
    # os.path.exists 函数判断文件夹是否存在
    folder = os.path.exists(path)

    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not folder:
        # os.makedirs 传入一个path路径，生成一个递归的文件夹；如果文件夹存在，就会报错,因此创建文件夹之前，需要使用os.path.exists(path)函数判断文件夹是否存在；
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print('文件夹创建成功：', path)

    else:
        print('文件夹已经存在：', path)
        
src_file = "test1.txt"
src_file_path = trace_base + "before_process/" + src_file
split_save = trace_base + "after_split/" + src_file.split(".txt")[0]

mkdir(split_save)

split = Split(src_file_path, split_save)

split.bylinecount(linecount = 750000) # 每个文件最多 750000 行
