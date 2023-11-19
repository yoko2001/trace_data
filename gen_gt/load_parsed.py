from trace_record import TraceRecord, load_str_record
import os

base = "/home/jl/trace_data/"
trace_base = base + "raw_traces/"
split_base = trace_base + "after_split/"

def scan_parsed_files(parsed_path):
    if not parsed_path.startswith(split_base):
        return None

    target = parsed_path
    filelist = os.listdir(target)
    allfiles = []
    for filename in filelist:
        filepath = os.path.join(target,filename)
        if filepath.endswith(".parsed"):
            allfiles.append(filepath)
    return allfiles

def load_single_file(filename):
    print("loading: ", filename)
    with open(filename,"r") as fin:
        while True:
            line = fin.readline()
            if (line == None):
                break
            record = load_str_record(line)
            print(record)
            
    

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
    files = scan_parsed_files(target)
    print(sorted(files))
    for file in files:
        load_single_file(file)
    