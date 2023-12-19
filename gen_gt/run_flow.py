import numpy as np
from ortools.graph.python import min_cost_flow
from trace_record import TraceRecord
from load_parsed import Record_Provider
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
label_file = None
result_file = None

class global_mapping(object):
    def __init__(self, input_flow):
        self.total_node_num = 0
        self.edges = [] # (start, end, cap, cost)
        self.sub_edges = []
        self.branch_edges_ind = []
        self.from_fake_src_edges_ind = []
        self.to_fake_tar_edges_ind = []
        self.current_v_index = 0
        self.total_v_num = 0
        self.raw_records = []
        self.label = []
        self.left = {}
        self.result_num = []
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
        
        if not ((len(self.result_num) ==self.current_v_index) and (len(self.result_num) == len(self.label))):
            raise RuntimeError("invalid 2 {0}!={1}!={2}".format(len(self.result_num), self.current_v_index, len(self.label)))

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
        self.label.extend([0]*4)
        self.result_num.extend([0]*4)
        
    def get_next_index(self):
        # alloc a new index for node
        self.current_v_index += 1
        return self.current_v_index - 1 

    def add_new_node(self, out, entry, record):
        main_line_last_v = self.last_v
        new_index = self.get_next_index()
        self.last_v = new_index
        self.index2record[new_index] = len(self.raw_records) #ind -> record_nid
        self.raw_records.append(record)
        self.label.append(0)
        self.result_num.append(0)
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
            # print("connect {0}->{1}".format(start_index, new_index))
            self.edges.append((start_index, new_index, capacity, cost))
            # assert(not (start_index + 1== new_index))            
        elif out == 0:   # 离开主干
            if (str(entry) in self.node_stacks.keys()):
                print("might have proplem:", record['process'], record['dir'], record['entry'], record['timestamp'])
            self.node_stacks[str(entry)] = new_index #overwrite if already stored
            #connect to main stream first
            self.edges.append((main_line_last_v, new_index, self.current_flow_max, 0))
        else:
            raise ValueError("can only use 0/1 node direction")

    def edge_is_from_fake_src(self, ind):
        return ind in self.from_fake_src_edges_ind
    
    def edge_is_to_fake_tar(self, ind):
        return ind in self.to_fake_tar_edges_ind
    
    def load_result(self, result):
        for (v_id, label) in result:
            self.label[v_id] += label
            rid = self.index2record[v_id]
            assert(self.raw_records[rid]['dir'] == 'e')
            self.result_num[v_id] += 1

    def finish_load_result(self):
        for i, (lable_sum, lable_num) in enumerate(zip(self.label, self.result_num)):
            if lable_num > 5:
                self.label[i] = lable_sum / lable_num
            else:
                self.label[i] = -1
            if not self.label[i] < 0.01:
                r_id = self.index2record[i]
                self.raw_records[r_id]['label'] = self.label[i]
                assert(self.raw_records[r_id]['dir'] == 'e')
        self.replay()
        self.record_dump()
    
    def replay(self):
        sub_ind_label = []
        #build the sub-map, sort them by label
        for i, label in enumerate(self.label):
            if label > -0.01:
                sub_ind_label.append((i, label))
                r_id = self.index2record[i]
                assert(self.raw_records[r_id]['dir'] == 'e')

        sub_input_flow = len(sub_ind_label) // 100 
        # inside sub_ind_label are evict point we can use
        # add those selected ev pts
        ind2subind = {}
        subind2ind = {}
        cur_subind = 2 #0 1 are spare
        inds = []
        sorted_ev_inds = []
        for (ind, label) in sub_ind_label:
            inds.append(cur_subind)
            sorted_ev_inds.append((cur_subind, label))
            ind2subind[ind] = cur_subind
            subind2ind[cur_subind] = ind
            cur_subind += 1
        num_sub_starts = cur_subind - 2
        print("num evict: ", num_sub_starts)
        print("num ev_inds: ", len(sorted_ev_inds))

        assert(num_sub_starts == len(sorted_ev_inds))
        assert(num_sub_starts == len(ind2subind.keys()))
        sorted_ev_inds = sorted(sorted_ev_inds, key=lambda x : x[1], reverse=True)
        # for ind, value in sorted_ev_inds:
        #     assert(ind in )
        #从大到小排序evict points，之后用这个顺序依次满足
        # add add paired refault of those selected ev pts
        substart2edge = {}
        num_sub_branch = 0
        for (start, end, cap, cost) in self.edges:
            if not start in ind2subind.keys(): #only use choosed starts
                continue
            if cost > -1: # mainline
                continue
            #start is choosed, add end also
            inds.append(cur_subind)
            ind2subind[end] = cur_subind
            subind2ind[cur_subind] = end
            num_sub_branch += 1
            #add the edge to subgraph
            subind_start = ind2subind[start]
            subind_end = cur_subind # ==ind2subind[end]
            self.sub_edges.append((subind_start, subind_end, 1 , -1, 0)) 
                                    # start        end       cap cost cur_flow
            assert(not subind_start in substart2edge.keys())
            substart2edge[subind_start] = len(self.sub_edges) - 1
            cur_subind += 1

        # we don't add no-start and no-end line to sub graph
        # we do the mainline rebuild
        print("num sub_edge: ", len(self.sub_edges))
        assert(len(sorted_ev_inds) == num_sub_branch)
        assert(len(substart2edge.keys()) == num_sub_branch)

        inds_to_rank = {}
        # 原始图中index与先后顺序保持相同的偏序关系，我们需要按照subind2ind[ind]对[2,cur_subid]进行排序
        inds_index = list(enumerate(inds))
        print("inds_index: {}".format(len(inds_index)))
        sorted_inds_index = sorted(inds_index, key = lambda x : subind2ind[x[1]], reverse=False)
        rank = [index for index,value in sorted_inds_index]
        inds_sorted = [value for index,value in sorted_inds_index]
        for i, ind in enumerate(inds):
            if not (inds[rank[i]] == inds_sorted[i]):
                print(i, rank[i], inds_sorted[rank[i]], ind)
            assert(ind not in inds_to_rank.keys())
            inds_to_rank[ind] = rank[i]
            assert(rank[i] <= len(inds))
        assert(len(inds) == len(sorted_inds_index) == len(inds_sorted) == len(inds_to_rank.keys()) == 2 * len(sorted_ev_inds))
        # now we can connect 0 - [inds] - 1
        mainline_edges = []
        last_mainline_pt = 0
        for subind in inds_sorted:
            mainline_edges.append(len(self.sub_edges))
            self.sub_edges.append((last_mainline_pt, subind, sub_input_flow, 0, sub_input_flow))
            last_mainline_pt = subind
        mainline_edges.append(len(self.sub_edges))
        self.sub_edges.append((last_mainline_pt, 1, sub_input_flow, 0, sub_input_flow))
        # print("assert {0} == {1} + {2}".format(len(self.sub_edges), len(mainline_edges), num_sub_branch))
        assert(len(self.sub_edges) == len(mainline_edges) + num_sub_branch)
        #now all edges are connected
        print("sub_input_flow: {}".format(sub_input_flow))
        for ev_ind in sorted_ev_inds:
            # 按照优先级顺序依次满足
            #find the sub_edge
            ev_ind = ev_ind[0] #get start ev_ind
            checkid = substart2edge[ev_ind]
            checkedge = self.sub_edges[checkid]
            re_ind = checkedge[1]
            assert(checkedge[2] == 1)
            # print(ev_ind, "->", re_ind)
            assert(ev_ind == checkedge[0])
            ev_index_in_ind = inds_to_rank[ev_ind]#rank[ev_ind]#inds.index(ev_ind)
            re_index_in_ind = inds_to_rank[re_ind]#rank[re_ind]#inds.index(re_ind)
            assert(re_ind > ev_ind)
            if not self.sub_edges[checkid][-1] > 0:
                #print("already not enough {}".format(ev_ind))
                continue
            #evind到reind之间的所有主干上的边都需要 -1
            #check first
            can_use = True
            down_edges = []
            for edge_id in mainline_edges:
                edge = self.sub_edges[edge_id]
                if edge[1] == 1 or edge[0] == 0:
                    continue
                start_in_ind = inds_to_rank[edge[0]]#rank[edge[0]]#inds.index(edge[0])
                end_in_ind = inds_to_rank[edge[1]]#rank[edge[1]]#inds.index(edge[1])
                
                if ev_index_in_ind < start_in_ind and end_in_ind < re_index_in_ind:
                    if self.sub_edges[edge_id][-1] > 0:
                        down_edges.append(edge_id)
                    else:
                        can_use = False
                        print(start_in_ind)
                        break
            if can_use and len(down_edges) > 0:
                print("can use {}".format(edge_id))
                for edge_id in down_edges:
                    edge = list(self.sub_edges[edge_id])
                    edge[-1] -= 1
                    edge = tuple(edge)
                    self.sub_edges[edge_id] = edge
                # flow !
                edge = list(self.sub_edges[checkid])
                edge[-1] += 1
                self.sub_edges[checkid] = edge
            else:
                print("can't use", checkid)

        #replay complete, now remap
        for ev_id in sorted_ev_inds: #check all ev_points
            edge_id = substart2edge[ev_id[0]]
            ori_ev_ind = subind2ind[ev_id[0]]
            self.left[ori_ev_ind] = self.sub_edges[edge_id][-1]
        
    def record_dump(self):
        global result
        record_label_index = 0
        _results = []
        accept_ev_rid = []
        accepted_left = []
        for ev_id in self.left.keys():
            accept_ev_rid.append(self.index2record[ev_id]) 
            accepted_left.append(self.left[ev_id])
        # for record_with_label in self.raw_records:
        for (rid, left) in zip(accept_ev_rid, accepted_left):
            record_with_label = self.raw_records[rid]
            if record_with_label == None:
                continue
            if not record_with_label['dir'] == 'e':
                continue
            
            #this is a evict point
            hist = record_with_label['se_hist']
            if hist==None:
                continue
            assert(len(hist) == 3)
            label = record_with_label['label']
            if label < 0.0:
                continue
            minseq = record_with_label['minseq']
            memcg_id = record_with_label['memcg_id']
            map(lambda x: minseq-x, hist)
            single_result = [record_label_index, label, memcg_id, left, hist] # [id ,lable, dist1, dist2, dist3]
            #我们无法使用left，这是由于left在实际运行时无法使用
            #但是我们通过不同的input_flow控制已经获得了资源量不同的时候的情况
            _results.append(single_result)
            record_label_index += 1

        with open(label_file, 'wb') as f:
            pickle.dump(_results, f)
            
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
        self.choosed_start_new_ind= []
        self.choosed_end_new_ind= []
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
            assert(ori_from < ori_to)
            choosed_ori_vertexs.append(ori_from)
            choosed_ori_vertexs.append(ori_to)
        choosed_ori_vertexs.sort() #all choosed ori inner vertex are sorted by time now
        main_stream_last = self.src_index
        assert(main_stream_last == 0)
        for ori_vertex in choosed_ori_vertexs:
            new_vertex = self.add_new_ori_vertex_ind(ori_vertex) #new index, mapping ok
            #adding these sorted vertexs to main stream
            self.sub_edges.append((main_stream_last, new_vertex, self.maxcache, 0))
            main_stream_last = new_vertex
        #connect to tar
        self.sub_edges.append((main_stream_last, self.tar_index, self.maxcache, 0))
        assert(len(self.sub_edges) == len(choosed_ori_edges) * 2 + 1)

        #collect all choosed edges
        for ori_edge in choosed_ori_edges:
            (ori_from, ori_to, _, _) = ori_edge
            assert(ori_to)
            new_from_ind = self.reverse_ind_mapping[ori_from]
            self.choosed_start_new_ind.append(new_from_ind)
            new_to_ind = self.reverse_ind_mapping[ori_to]
            self.choosed_end_new_ind.append(new_to_ind)
            assert(new_from_ind < new_to_ind and ori_from < ori_to)
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
        self.vertex_cost[str(self.fake_src_index)] = min(self.no_start_num, self.maxcache - self.input_flow - 5) # 5 is for safty 
        total_input = self.vertex_cost[str(self.src_index)] + self.vertex_cost[str(self.fake_src_index)]
        #the comsume of fake_tar can be a little bit less than 100%
        #we can give 0-20% loss on that
        self.vertex_cost[str(self.fake_tar_index)] = -self.no_end_num
        self.vertex_cost[str(self.tar_index)] = -(total_input - self.no_end_num)
    
    def check_node_cost(self):    
        print("costs:", self.vertex_cost, "; max:", self.maxcache, "; v_num: ", len(self.reverse_ind_mapping.keys()), "; br_num: ", self.branched_edge_num)

global_results = []
global_graph = None
def run_a_sample(max_edge, maxcache, lock, seed):
    global global_results
    #generate a new sample and run it
    mapping = global_graph
    time_begin = time.time()
    sub_graph = state(mapping=mapping, seed=seed, max_edge=max_edge, maxcache=maxcache)
    time_end = time.time()
    print("建图与建立映射耗时："+str(time_end-time_begin)+"s")

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
    print("流计算耗时："+str(time_end-time_begin)+"s")

    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        sub_graph.check_node_cost()
        return None
    solution_flows = smcf.flows(all_arcs)
    costs = solution_flows * unit_costs
    result = []
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        new_from = smcf.tail(arc)
        new_to = smcf.head(arc)
        if (new_from < 4) or (new_to < 4): #virtual nodes
            continue
        if (new_from not in sub_graph.choosed_start_new_ind):
            continue
        if (new_to not in sub_graph.choosed_end_new_ind):
            continue
        ori_to = sub_graph.ind_mapping[new_to]
        ori_from = sub_graph.ind_mapping[new_from]
        assert((ori_from < ori_to) == (new_from < new_to) == True)
        if flow == 1:
            result.append((ori_from, 1))
        else:
            result.append((ori_from, 0))
    return result

from tqdm import tqdm

if __name__ == "__main__":
    max_cache_choises = [1000, 2000, 4000]
    small = False#True # True
    if len(sys.argv) > 1:
        if not len(sys.argv) == 4:
            raise NotImplementedError("can only have one argument", sys.argv)
        src_file_name = sys.argv[-3]
        max_cache = int(sys.argv[-2])
        runs = int(sys.argv[-1])
        if not src_file_name.endswith(".txt"):
            src_name = src_file_name.split(".txt")[0]
        else:
            src_name = src_file_name
    else:
        src_name = "test1"
        max_cache = 2000
        runs = 500
    
    if small:
        max_cache = 500
        runs = 200

    target_base = os.path.join(split_base, src_name)
    
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_file = os.path.join(
        split_base, 
        '{0}_{1}_{2}_{3}_result.pkl'.format(src_name, timenow, str(max_cache), str(runs))
    )
    label_file = os.path.join(
        split_base, 
        '{0}_{1}_{2}_{3}_label.pkl'.format(src_name, timenow, str(max_cache), str(runs))
    )
    print(target_base)

    loader = Record_Provider(target_base)
    
    mapping = global_mapping(max_cache)     #1000 = 4mb
    count = 0
    while(True):
        record = loader.next_record()
        count += 1
        if small and count >= 10000000:
            break
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
    
    runs_seeds = None
    process_pool = None
    pfunc = None

    total_run = runs

    if small:
        runs_seeds = np.arange(total_run).tolist()
        process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-3)
        # 定义偏函数，并传入固定参数
        pfunc = partial(run_a_sample, 50000, 120, lock)
    else:
        runs_seeds = np.arange(total_run).tolist()

        for i in range(0, len(runs_seeds)):
            runs_seeds[i] += random.randint(12345,67890)
        process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-3)
        # 定义偏函数，并传入固定参数
        pfunc = partial(run_a_sample, 100000, 250, lock)

    # 执行map，传入seeds
    results = process_pool.map(pfunc, runs_seeds)
    # results = tqdm(process_pool.map(pfunc, runs_seeds), total=total_run, desc='采样进度')
    results = tqdm(process_pool.imap(func=pfunc, iterable=runs_seeds), total=total_run, desc='采样进度')
    assert(len(results) == len(runs_seeds))
    for result in results:
        if result:
            mapping.load_result(result)
    process_pool.close()
    mapping.finish_load_result()
    # with open(result_file, 'wb') as f:
    #     pickle.dump(results, f)