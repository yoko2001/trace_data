import numpy as np
from ortools.graph.python import min_cost_flow
from trace_record import TraceRecord
from load_parsed import Record_Provider
import joblib
import time
import sys
import os
# Instantiate a SimpleMinCostFlow solver.
smcf = min_cost_flow.SimpleMinCostFlow()

# Define four parallel arrays: sources, destinations, capacities,
# and unit costs between each pair. For instance, the arc from node 0
# to node 1 has a capacity of 15.

def gen_start_end_nodes():
    return None, None

# start_nodes = np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 2, 9, 4, 10])
# end_nodes =   np.array([1, 2, 3, 4, 5, 6, 7, 8, 6, 9, 3, 10, 5])
# capacities = np.array( [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1])
# unit_costs = np.array( [0, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0,-1, 0])
# # Define an array of supplies at each node.
# supplies =             [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]

# # Add arcs, capacities and costs in bulk using numpy.
# all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
#     start_nodes, end_nodes, capacities, unit_costs
# )

# # Add supply for each nodes.
# smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

# # Find the min cost flow.
# status = smcf.solve()

# if status != smcf.OPTIMAL:
#     print("There was an issue with the min cost flow input.")
#     print(f"Status: {status}")
#     exit(1)
# print(f"Minimum cost: {smcf.optimal_cost()}")
# print("")
# print(" Arc    Flow / Capacity Cost")
# solution_flows = smcf.flows(all_arcs)
# costs = solution_flows * unit_costs
# for arc, flow, cost in zip(all_arcs, solution_flows, costs):
#     print(
#         f"{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}"
#     )

filepwd = os.path.realpath(__file__)
trace_preprocesspwd = os.path.dirname(filepwd)
base = os.path.dirname(trace_preprocesspwd)
trace_base = os.path.join(base , "raw_traces/")
split_base = os.path.join(trace_base, "after_split/")

if __name__ == "__main__":
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
    # target = "D:\\yoko\\trace_data\\raw_traces\\after_split\\test1"
    print(target_base)

    loader = Record_Provider(target_base)

    num = 0
    basic_id = 1
    start_nodes = []
    end_nodes =  []
    capacities = []
    unit_costs = []
    entry_id = {}
    max_cache = 2000
    while(True):
        record = loader.next_record()
        if record == None:
            print(num)
            break
                   
        entry = record['entry']
        in_out = record['dir']
        
        if in_out == 'e':
            entry_id[entry] = basic_id
            # add slow edge
            start_nodes.append(basic_id - 1)
            end_nodes.append(basic_id)
            capacities.append(max_cache)
            unit_costs.append(0)
        elif in_out == 'r':
            if entry in entry_id.keys():
                # add fast edge
                start_nodes.append(entry_id[entry])
                end_nodes.append(basic_id)
                capacities.append(1)
                unit_costs.append(-1)
                
                # add slow edge
                start_nodes.append(basic_id - 1)
                end_nodes.append(basic_id)
                capacities.append(max_cache)
                unit_costs.append(0)
                
                del entry_id[entry]
            else:
                max_cache += 1
                continue
                
        num += 1
        basic_id += 1

    # add slow edge
    start_nodes.append(basic_id - 1)
    end_nodes.append(basic_id)
    capacities.append(max_cache)
    unit_costs.append(0)
    for entry in entry_id.keys():
        # add fast edge
        start_nodes.append(entry_id[entry])
        end_nodes.append(basic_id)
        capacities.append(1)
        unit_costs.append(-1)
    for i in range(len(capacities)):
        if capacities[i] != 1:
            capacities[i] = max_cache
    print("max_cache:"+str(max_cache))
    print("node_num:"+str(len(set(start_nodes))))
    print(len(start_nodes))
    print(len(end_nodes))
    print(len(capacities))
    print(len(unit_costs))
    
    start_nodes = np.array(start_nodes)
    end_nodes = np.array(end_nodes)
    capacities = np.array(capacities)
    unit_costs = np.array(unit_costs)
    
    supplies = list(np.zeros(len(set(start_nodes))+1, dtype=int))
    supplies[0] = max_cache
    supplies[-1] = -max_cache
    # print(supplies)


    # Add arcs, capacities and costs in bulk using numpy.
    smcf = min_cost_flow.SimpleMinCostFlow()
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
    start_nodes, end_nodes, capacities, unit_costs
    )

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    time_begin = time.time()
    status = smcf.solve()
    time_end = time.time()
    print("耗时："+str(time_end-time_begin)+"s")
    joblib.dump(status, 'result_status.pkl')

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        exit(1)
    print(f"Minimum cost: {smcf.optimal_cost()}")
    print("")
    print(" Arc    Flow / Capacity    Cost")
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs