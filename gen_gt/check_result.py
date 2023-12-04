import pickle

file = "/home/jl/trace_data/raw_traces/after_split/small1_21:10:28_1000_result.pkl"
file = "/home/jl/trace_data/raw_traces/after_split/small2_02:54:57_1000_result.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)
    print(len)