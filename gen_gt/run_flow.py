import numpy as np
from ortools.graph.python import min_cost_flow
from trace_record import TraceRecord
from load_parsed import Record_Provider
import joblib
import time
import sys
import os
import random
import math
import multiprocessing
from functools import partial
import pickle
import datetime

filepwd = os.path.realpath(__file__)
trace_preprocesspwd = os.path.dirname(filepwd)
base = os.path.dirname(trace_preprocesspwd)
trace_base = os.path.join(base , "raw_traces/")
split_base = os.path.join(trace_base, "after_split/")
    

class global_mapping(object):
    def __init__(self, input_flow):
        self.total_node_num = 0
        self.edges = [] # (start, end, cap, cost)
        self.branch_edges_ind = []
        self.from_fake_src_edges_ind = []
        self.to_fake_tar_edges_ind = []
        self.current_v_index = 0
        self.total_v_num = 0
        self.raw_records = []
        self.index2record = {}
        self.node_stacks = {}
        self.input_flow = input_flow
        self.current_flow_max = input_flow
        self.last_v = -1
        self.src_index = -1
        self.fake_src_index = -1
        self.tar_index = -1
        self.fake_tar_index = -1
        self.vertex_cost = {}
        self.prepare_accept_normal_vertexes()
        self.no_start_num = 0
        self.no_end_num = 0
        self.complete_num = 0
    
    def close_accept_normal_vertexes(self):
        #connect last normal node to target
        self.edges.append((self.last_v, self.tar_index, self.current_flow_max, 0))

        for entry in self.node_stacks.keys():
            start_index = self.node_stacks[entry]
            self.no_end_num += 1
            self.to_fake_tar_edges_ind.append(len(self.edges))
            #connect these node to virtual end
            self.edges.append((start_index, self.fake_tar_index, 1, -1))
        
        #safety check
        if not len(self.edges) == self.current_v_index - 3 + self.no_end_num + self.no_start_num + self.complete_num:
            print(len(self.edges), self.current_v_index, self.no_end_num, self.no_start_num, self.complete_num)
            raise RuntimeError("invalid graph")
        
    def reset_tar_vertex(self):
        #set vertex_cost
        self.vertex_cost[self.src_index] = self.input_flow
        self.vertex_cost[self.fake_src_index] = self.no_start_num
        total_input = self.input_flow + self.no_start_num
        #the comsume of fake_tar can be a little bit less than 100%
        #we can give 0-20% loss on that
        into_fake_tar = math.ceil(self.no_end_num * (1.0 - random.random() * 0.2))
        self.vertex_cost[self.fake_tar_index] = -into_fake_tar
        self.vertex_cost[self.tar_index] = -(total_input - into_fake_tar)
        
    def prepare_accept_normal_vertexes(self):
        #add the 1sr virtual v
        #init src_index 用于供给主干起始点的流量的虚拟点
        new_index = self.get_next_index()
        self.src_index = new_index 
        self.vertex_cost[str(self.src_index)] = 0
        self.index2record[new_index] = len(self.raw_records)
        self.raw_records.append(None)
        
        #add the 2nd virtual v
        #init fake_src_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.get_next_index()
        self.fake_src_index = new_index
        self.vertex_cost[str(self.fake_src_index)] = 0
        self.index2record[new_index] = len(self.raw_records)
        self.raw_records.append(None)
        
        #add the 3rd virtual v
        #init tar_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.get_next_index()
        self.tar_index = new_index
        self.vertex_cost[str(self.tar_index)] = 0
        self.index2record[new_index] = len(self.raw_records)
        self.raw_records.append(None)
        
        #add the 4th virtual v
        #init fake_src_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.get_next_index()
        self.fake_tar_index = new_index
        self.vertex_cost[str(self.fake_tar_index)] = 0
        self.index2record[new_index] = len(self.raw_records)
        self.raw_records.append(None)
        
        self.last_v = self.src_index
        
    def get_next_index(self):
        # alloc a new index for node
        self.current_v_index += 1
        return self.current_v_index - 1 

    def add_new_node(self, out, entry, record):
        main_line_last_v = self.last_v
        new_index = self.get_next_index()
        self.last_v = new_index
        self.index2record[new_index] = len(self.raw_records)
        self.raw_records.append(record)
        if out == 1: # 汇入主干
            #connect to main stream first
            self.edges.append((main_line_last_v, new_index, self.current_flow_max, 0))
            cost = -1
            capacity = 1
            if str(entry) in self.node_stacks.keys():
                #has start point
                self.complete_num += 1
                start_index = self.node_stacks[str(entry)]
                del self.node_stacks[str(entry)]
                self.branch_edges_ind.append(len(self.edges)) #record edge ind
            else:
                #has no start point
                self.no_start_num += 1
                start_index = self.fake_src_index    
                self.from_fake_src_edges_ind.append(len(self.edges))
            #connect start / fake_src with this end node
            self.edges.append((start_index, new_index, capacity, cost))
            
        elif out == 0:   # 离开主干
            self.node_stacks[str(entry)] = new_index
            #connect to main stream first
            self.edges.append((main_line_last_v, new_index, self.current_flow_max, 0))
        else:
            raise ValueError("can only use 0/1 node direction")

    def edge_is_from_fake_src(self, ind):
        return ind in self.from_fake_src_edges_ind
    
    def edge_is_to_fake_tar(self, ind):
        return ind in self.to_fake_tar_edges_ind
    
class state(object):
    def __init__(self, mapping, seed=1234, max_edge=300000, maxcache=500):
        random.seed(seed)
        self.mapping = mapping
        self.max_branch_edge = min(max_edge, len(self.mapping.branch_edges_ind))
        self.maxcache = max(maxcache, int(mapping.current_flow_max * self.max_branch_edge / len(mapping.branch_edges_ind)  *2))
        self.input_flow = self.maxcache // 2
        # print("substate maxcache {0}, maxedge {1}".format(self.maxcache, self.max_branch_edge))
        self.ind_mapping = {}         #sub->ori
        self.reverse_ind_mapping = {} #ori->sub
        self.sub_ind_cur = 0
        self.sub_edges = [] # (start, end, cap, cost)
        self.branched_edge_num = 0
        self.no_start_num = 0
        self.no_end_num = 0
        self.vertex_cost = {}
        self.src_index = -1
        self.fake_src_index = -1
        self.tar_index = -1
        self.fake_tar_index = -1
        self.get_sub_mapping() #obtain sub graph
        #safety check
        if not len(self.sub_edges) == len(self.ind_mapping.keys()) - 3 + self.no_end_num + self.no_start_num + self.branched_edge_num:
            print(len(self.sub_edges), len(self.ind_mapping.keys()), self.no_end_num, self.no_start_num, self.branched_edge_num)
            raise RuntimeError("invalid graph")
    def prepare_accept_normal_vertexes(self):
        #add the 1sr virtual v
        #init src_index 用于供给主干起始点的流量的虚拟点
        new_index = self.add_new_ori_vertex_ind(0)
        self.src_index = new_index 
        self.vertex_cost[str(self.src_index)] = 0
        #add the 2nd virtual v
        #init fake_src_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.add_new_ori_vertex_ind(1)
        self.fake_src_index = new_index
        self.vertex_cost[str(self.fake_src_index)] = 0
        #add the 3rd virtual v
        #init tar_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.add_new_ori_vertex_ind(2)
        self.tar_index = new_index
        self.vertex_cost[str(self.tar_index)] = 0
        
        #add the 4th virtual v
        #init fake_src_index 用于供给所有无起始点的直流流量的虚拟点
        new_index = self.add_new_ori_vertex_ind(3)
        self.fake_tar_index = new_index
        self.vertex_cost[str(self.fake_tar_index)] = 0
        assert(new_index == 3)
        
    def add_new_ori_vertex_ind(self, ori_vertex):
        sub_ind = self.sub_ind_cur
        assert(sub_ind not in self.ind_mapping.keys())
        assert(ori_vertex not in self.reverse_ind_mapping.keys())
        self.ind_mapping[sub_ind] = ori_vertex
        self.reverse_ind_mapping[ori_vertex]=sub_ind
        self.sub_ind_cur+=1 #next ind
        return sub_ind
    
    def get_sub_mapping(self):
        self.prepare_accept_normal_vertexes()
        choosed_edges = random.sample(self.mapping.branch_edges_ind, self.max_branch_edge) #choosed edges
        choosed_ori_edges = []
        for ori_edge_ind in choosed_edges:
            ori_edge = self.mapping.edges[ori_edge_ind] 
            choosed_ori_edges.append(ori_edge)
        #sort using ori_from
        choosed_ori_edges.sort(key=lambda x:x[0])
        choosed_ori_vertexs = []
        for ori_edge in choosed_ori_edges:
            #一定是中间节点，ori_from, ori_to, cap, cost=-1;且不会重复
            (ori_from, ori_to, _, _) = ori_edge
            choosed_ori_vertexs.append(ori_from)
            choosed_ori_vertexs.append(ori_to)
        choosed_ori_vertexs.sort() #all choosed ori inner vertex are sorted by time now
        main_stream_last = self.src_index
        for ori_vertex in choosed_ori_vertexs:
            new_vertex = self.add_new_ori_vertex_ind(ori_vertex) #new index, mapping ok
            #adding these sorted vertexs to main stream
            self.sub_edges.append((main_stream_last, new_vertex, self.maxcache, 0))
            main_stream_last = new_vertex
        #connect to tar
        self.sub_edges.append((main_stream_last, self.tar_index, self.maxcache, 0))

        #collect all choosed edges
        for ori_edge in choosed_ori_edges:
            (ori_from, ori_to, _, _) = ori_edge
            new_from_ind = self.reverse_ind_mapping[ori_from]
            new_to_ind = self.reverse_ind_mapping[ori_to]
            self.sub_edges.append((new_from_ind, new_to_ind, 1, -1))
            self.branched_edge_num += 1
        
        #collect all ori_from_tar
        for ori_vertex in choosed_ori_vertexs:
            if self.mapping.edge_is_from_fake_src(ori_vertex):
                #connect it to fake src
                new_to = self.reverse_ind_mapping[ori_vertex]
                self.sub_edges.append((self.fake_src_index, new_to, 1, 0))
                self.no_start_num += 1
            if self.mapping.edge_is_to_fake_tar(ori_vertex):
                #connect it to fake tar
                new_from = self.reverse_ind_mapping[ori_vertex]
                self.sub_edges.append((new_from, self.fake_tar_index, 1, 0))
                self.no_end_num += 1
        
        #set all costs
        self.vertex_cost[str(self.src_index)] = self.input_flow
        self.vertex_cost[str(self.fake_src_index)] = self.no_start_num
        total_input = self.input_flow + self.no_start_num
        #the comsume of fake_tar can be a little bit less than 100%
        #we can give 0-20% loss on that
        self.vertex_cost[str(self.fake_tar_index)] = -self.no_end_num
        self.vertex_cost[str(self.tar_index)] = -(total_input - self.no_end_num)
        # print(self.vertex_cost)

global_results = []
global_graph = None

def run_a_sample(max_edge, maxcache, lock, seed):
    global global_results
    #generate a new sample and run it
    mapping = global_graph
    sub_graph = state(mapping=mapping, seed=seed, max_edge=max_edge, maxcache=maxcache)
    start_nodes = []
    end_nodes =  []
    capacities = []
    unit_costs = []
    for edge in sub_graph.sub_edges:
        (start, end, cap, cost) = edge
        assert(cap > 0)
        assert(cost <= 0)
        start_nodes.append(start)
        end_nodes.append(end)
        capacities.append(cap)
        unit_costs.append(cost)
        
    assert(len(start_nodes) == len(end_nodes))
    assert(len(capacities) == len(unit_costs))
    
    start_nodes = np.array(start_nodes)
    end_nodes = np.array(end_nodes)
    capacities = np.array(capacities)
    unit_costs = np.array(unit_costs)
    supplies = [0] * sub_graph.sub_ind_cur
    for new_ind in sub_graph.vertex_cost.keys():
        supplies[int(new_ind)] = int(sub_graph.vertex_cost[new_ind])

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

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        return
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    result = []
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        new_from = smcf.tail(arc)
        if (arc < 4): #actual nodes
            continue
        new_to = smcf.head(arc)
        ori_from = sub_graph.ind_mapping[new_from]
        if flow == 1:
            result.append((ori_from, 1))
        else:
            result.append((ori_from, 0))
    return result

if __name__ == "__main__":
    max_cache_choises = [1000, 2000, 4000]
    if len(sys.argv) > 1:
        if not len(sys.argv) == 3:
            raise NotImplementedError("can only have one argument")
        src_file_name = sys.argv[-2]
        max_cache = int(sys.argv[-1])
        if not src_file_name.endswith(".txt"):
            src_name = src_file_name.split(".txt")[0]
        else:
            src_name = src_file_name
    else:
        src_name = "test1"
        max_cache = 2000
    target_base = os.path.join(split_base, src_name)
    
    timenow = datetime.datetime.now().strftime('%H:%M:%S')

    result_file = os.path.join(split_base, src_name +'_' + timenow + '_' + str(max_cache) +'_result.pkl')
    print(target_base)

    loader = Record_Provider(target_base)
    
    mapping = global_mapping(max_cache)     #1000 = 4mb
    count = 0
    while(True):
        record = loader.next_record()
        count += 1
        if record == None:
            mapping.close_accept_normal_vertexes()
            break
        entry = record['entry']
        in_out = record['dir']
        
        if in_out == 'r':
            mapping.add_new_node(1, entry, record)
        elif in_out == 'e':
            mapping.add_new_node(0, entry, record)
        else:
            raise ValueError("got broken record")
    global_graph = mapping
    # run_a_sample(mapping, 12, 50000, 200, None)
    lock = multiprocessing.Manager().Lock()
    
    runs_seeds = np.arange(50).tolist()
    process_pool = multiprocessing.Pool(processes=30)
    # 定义偏函数，并传入固定参数
    pfunc = partial(run_a_sample, 300000, 200, lock)

    # 执行map，传入seeds
    results = process_pool.map(pfunc, runs_seeds)
    assert(len(results) == len(runs_seeds))
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)