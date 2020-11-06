import sys
import time
import numpy as np 
import multiprocessing 
from multiprocessing import Pool

class AllAgents:
    def __init__(self, od_df=None, network=None):
        self.agents = dict()
        self.network = network

        for a in od_df.itertuples():
            self.agents[getattr(a, 'agent_id')] = Agent(getattr(a, 'agent_id'), getattr(a, 'origin_nid'), getattr(a, 'destin_nid'), getattr(a, 'dept_time'), network)

class Agent:
    def __init__(self, aid, origin_nid, destin_nid, dept_time, network):
        #input
        self.id = aid
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
        # self.network = network
        ### derived
        self.cls = 'vn{}'.format(self.origin_nid) # current link start node
        self.cle = self.origin_nid # current link end node
        self.origin_idsp = self.origin_nid + 1
        self.destin_idsp = self.destin_nid + 1
        ### Empty
        self.route_igraph = []
        self.find_route = None
        self.status = None
        self.cl_enter_time = None

    def load_trip(self, t_now=None):
        if (self.dept_time == t_now):
            initial_edge = self.network.node2link_dict[self.route_igraph[0]]
            self.network.links[initial_edge].run_veh.append(self.id)
            self.status = 'loaded'
            self.cl_enter_time = t_now

    def prepare_agent(self, node_id=None):
        assert self.cle == node_id, "agent next node {} is not the transferring node {}, route {}".format(self.cle, node_id, self.route_igraph)
        if self.destin_nid == node_id: ### current node is agent destination
            return None, None, 0 ### id, next_node, dir
        agent_next_node = [end for (start, end) in self.route_igraph if start == node_id][0]
        ol = self.network.node2link_dict[(node_id, agent_next_node)]
        x_start, y_start = self.network.nodes[self.cls].lon, self.network.nodes[self.cls].lat
        x_mid, y_mid = self.network.nodes[node_id].lon, self.network.nodes[node_id].lat
        x_end, y_end = self.network.nodes[agent_next_node].lon, self.network.nodes[agent_next_node].lat
        in_vec, out_vec = (x_mid-x_start, y_mid-y_start), (x_end-x_mid, y_end-y_mid)
        dot, det = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]), (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
        agent_dir = np.arctan2(det, dot)*180/np.pi 
        return agent_next_node, ol, agent_dir
    
    def move_agent(self, t_now, new_cls, new_cle, new_status, il, ol, transfer_s=None, transfer_e=None):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now
        #if self.id==349: print(t_now, self.cls, self.cle, self.status)
        ### pass key location
        if (il==transfer_s) and (ol==transfer_e): return 1
        else: return 0
    
    def get_path(self):
        # g = self.network.g
        sp = g.dijkstra(self.cle+1, self.destin_idsp)
        sp_dist = sp.distance(self.destin_idsp)

        if sp_dist > 10e7:
            self.route_igraph = []
            self.find_route = 'n_a'
            sp.clear()
        else:
            sp_route = sp.route(self.destin_idsp)
            self.route_igraph = [(self.cls, self.cle)] + [(start_sp-1, end_sp-1) for (start_sp, end_sp) in sp_route]
            self.find_route = 'a'
            sp.clear()
