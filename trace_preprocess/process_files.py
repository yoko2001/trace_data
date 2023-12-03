import os
import sys
from process_single_file import parse_file_by_line
import multiprocessing

filepwd = os.path.realpath(__file__)
trace_preprocesspwd = os.path.dirname(filepwd)
base = os.path.dirname(trace_preprocesspwd)
trace_base = os.path.join(base , "raw_traces/")
split_base = os.path.join(trace_base, "after_split/")


def scanfile(path):
    filelist = os.listdir(path)
    allfiles = []
    for filename in filelist:
        filepath = os.path.join(path,filename)
        if filepath.endswith(".txt"):
            allfiles.append(filepath)
    return allfiles

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            raise NotImplementedError("can only have one argument")
        src_file_name = sys.argv[-1]
        if not src_file_name.endswith(".txt"):
            src_name = src_file_name.split(".txt")[0]
        else:
            src_name = src_file_name
    else:
        src_name = "test1"
    target = src_name
    target_dir = os.path.join(split_base , target)
    print(target_dir)
    filelist = scanfile(target_dir)
    
    pool = multiprocessing.Pool(processes = min(len(filelist), 8)) # max 8 cores
    
    for file in filelist:
        target = file + '.parsed'
        print("{0} => {1}".format(file, target))
        pool.apply_async(parse_file_by_line, (file, target, ))
        #parse_file_by_line(file, target)
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
