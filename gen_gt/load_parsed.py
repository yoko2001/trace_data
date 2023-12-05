from trace_record import TraceRecord, load_str_record
import os
import sys

filepwd = os.path.realpath(__file__)
trace_preprocesspwd = os.path.dirname(filepwd)
base = os.path.dirname(trace_preprocesspwd)
trace_base = os.path.join(base , "raw_traces/")
split_base = os.path.join(trace_base, "after_split/")

def scan_parsed_files(parsed_path):
    if not parsed_path.startswith(split_base):
        return None

    target = parsed_path
    filelist = os.listdir(target)
    # print(fi)
    allfiles = []
    for filename in filelist:
        filepath = os.path.join(target,filename)
        if filepath.endswith(".parsed"):
            allfiles.append(filepath)
    return allfiles

def load_single_file(filename):
    print("loading: ", filename)
    count = 0
    with open(filename,"r") as fin:
        while True:
            line = fin.readline()
            if (line == None):
                break
            if (len(line) < 20):
                print(line)
                break
            record = load_str_record(line)
            count+=1

class Record_Provider(object):
    def __init__(
        self,  
        target_dir
    ):
        self.files = scan_parsed_files(target_dir)
        self.files = sorted(self.files) #get all files to load
        self.filenum = len(self.files)
        self.cur_file_num = 0
        self.fin_cur = None
        self.establish_flow(0)
    
    def establish_flow(self, index):
        if not self.fin_cur == None:
            self.fin_cur.close()
        self.fin_cur = open(self.files[index],"r")
    
    def close_flow(self):
        if not self.fin_cur == None:
            self.fin_cur.close()
    
    def next_record(self):
        while True:
            line = self.fin_cur.readline()
            if (line == None or len(line) < 10):
                self.cur_file_num += 1
                if self.cur_file_num < self.filenum:
                    self.establish_flow(self.cur_file_num) #move to next
                else:
                    return None
            else:
                record = load_str_record(line)
                return record



if __name__ == '__main__':
    target = "/home/jl/trace_data/raw_traces/after_split/test1"
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
    target_base = os.path.join(split_base, src_name)

    files = scan_parsed_files(target_base)
    files = sorted(files)
    print(files)
    for file in files:
        load_single_file(file)
    