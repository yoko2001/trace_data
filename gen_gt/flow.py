import numpy as np
from ortools.graph.python import min_cost_flow
from trace_record import TraceRecord
from load_parsed import Record_Provider
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
    
if __name__ == "__main__":
    target = "/home/jl/trace_data/raw_traces/after_split/test1"
    loader = Record_Provider(target)
    num = 0
    while(True):
        record = loader.next_record()
        if record == None:
            print(num)
            exit(0)
        num += 1
        