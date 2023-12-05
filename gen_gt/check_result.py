import pickle
import os
import sys
filepwd = os.path.realpath(__file__)
trace_preprocesspwd = os.path.dirname(filepwd)
base = os.path.dirname(trace_preprocesspwd)
trace_base = os.path.join(base , "raw_traces/")
split_base = os.path.join(trace_base, "after_split/")

allfiles=os.listdir(split_base)
filelist = []
for file in allfiles:
    if file.endswith("label.pkl"):
        filelist.append(file)
print(filelist)
sorted(filelist)
for file in filelist:
    file_path = os.path.join(split_base, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(file_path, len(data))
        total_scan = 0
        num_00_01 = 0
        num_01_02 = 0
        num_02_03 = 0
        num_03_04 = 0
        num_04_05 = 0
        num_05_06 = 0
        num_06_07 = 0
        num_07_08 = 0
        num_08_10 = 0

        for line in data:
            if line[1] < 0.1:
                num_00_01 += 1
            elif line[1] < 0.2:
                num_01_02 += 1
            elif line[1] < 0.3:
                num_02_03 += 1
            elif line[1] < 0.4:
                num_03_04 += 1
            elif line[1] < 0.5:
                num_04_05 += 1
            elif line[1] < 0.6:
                num_05_06 += 1
            elif line[1] < 0.7:
                num_06_07 += 1
            elif line[1] < 0.8:
                num_07_08 += 1
            elif line[1] < 1.1:
                num_08_10 += 1
        print("[0-0.1] num: {}".format(num_00_01) )
        print("[0.1-0.2] num: {}".format(num_01_02) )
        print("[0.2-0.3] num: {}".format(num_02_03))
        print("[0.3-0.4] num: {}".format(num_03_04))
        print("[0.4-0.5] num: {}".format(num_04_05))
        print("[0.5-0.6] num: {}".format(num_05_06))
        print("[0.6-0.7] num: {}".format(num_06_07))
        print("[0.7-0.8] num: {}".format(num_07_08))
        print("[0.8-1] num: {}".format(num_08_10))

        assert(
            num_08_10 +num_07_08 + num_06_07 + num_05_06 + num_04_05 +
            num_03_04 + num_02_03 + num_01_02 + num_00_01 
            == 
            len(data)
        )