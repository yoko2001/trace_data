import os
from process_single_file import parse_file_by_line
base = "/home/jl/trace_data/"
trace_base = base + "raw_traces/"
split_base = trace_base + "after_split/"

target = "test1"
target_dir = split_base + target + "/"

def scanfile(path):
    filelist = os.listdir(path)
    allfiles = []
    for filename in filelist:
        filepath = os.path.join(path,filename)
        if filepath.endswith(".txt"):
            allfiles.append(filepath)
    return allfiles

filelist = scanfile(target_dir)

for file in filelist:
    target = file + '.parsed'
    parse_file_by_line(file, target)