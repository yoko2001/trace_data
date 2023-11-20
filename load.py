import joblib
from ortools.graph.python import min_cost_flow
smcf = min_cost_flow.SimpleMinCostFlow()
status = joblib.load('result_status.pkl')
if status != smcf.OPTIMAL:
    print("There was an issue with the min cost flow input.")
    print(f"Status: {status}")
    exit(1)
print(f"Minimum cost: {smcf.optimal_cost()}")
print("")
print(" Arc    Flow / Capacity    Cost")
# solution_flows = smcf.flows(all_arcs)
# costs = solution_flows * unit_costs