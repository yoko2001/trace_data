import pickle

file = "/home/jl/trace_data/raw_traces/after_split/small1_result.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)