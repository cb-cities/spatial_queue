import os
import gc
import sys
import time 
import random 
import numpy as np
import pandas as pd 
from ctypes import *
import scipy.io as sio
from shapely.wkt import loads
import scipy.sparse as ssparse
from operator import itemgetter
from multiprocessing import Pool, Process, Queue
from pympler.asizeof import asizeof

absolute_path = '/home/bingyu/Documents/spatial_queue'
sys.path.insert(0, absolute_path+'/../')
from sp import interface 

class Node:
    def __init__(self, id, lon, lat, type, osmid=None):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.type = type
        self.osmid = osmid
        ### derived
        # self.id_sp = self.id + 1
        self.in_links = {} ### {in_link_id: straight_ahead_out_link_id, ...}
        self.out_links = []
        self.go_vehs = [] ### veh that moves in this time step
        self.status = None

    def create_virtual_node(self):
        return Node('vn{}'.format(self.id), self.lon+0.001, self.lat+0.001, 'v')

    def create_virtual_link(self):
        return Link('n{}_vl'.format(self.id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.id), self.id, 'LINESTRING({} {}, {} {})'.format(self.lon+0.001, self.lat+0.001, self.lon, self.lat))
    
    def calculate_straight_ahead_links(self):
        for il in self.in_links.keys():
            x_start = node_id_dict[link_id_dict[il].start_nid].lon
            y_start = node_id_dict[link_id_dict[il].start_nid].lat
            in_vec = (self.lon-x_start, self.lat-y_start)
            sa_ol = None ### straight ahead out link
            ol_dir = 180
            for ol in self.out_links:
                x_end = node_id_dict[link_id_dict[ol].end_nid].lon
                y_end = node_id_dict[link_id_dict[ol].end_nid].lat
                out_vec = (x_end-self.lon, y_end-self.lat)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                new_ol_dir = np.arctan2(det, dot)*180/np.pi
                if abs(new_ol_dir)<ol_dir:
                    sa_ol = ol
                    ol_dir = new_ol_dir
            if (abs(ol_dir)<=45) and link_id_dict[il].type=='real':
                self.in_links[il] = sa_ol

    def find_go_vehs(self, go_link):
        go_vehs_list = []
        x_mid, y_mid = self.lon, self.lat
        x_start, y_start = node_id_dict[go_link.start_nid].lon, node_id_dict[go_link.start_nid].lat
        in_vec = (x_mid-x_start, y_mid-y_start)
        incoming_lanes = int(np.floor(go_link.lanes))
        incoming_vehs = len(go_link.queue_veh)
        for ln in range(min(incoming_lanes, incoming_vehs)):
            agent_id = go_link.queue_veh[ln]
            agent_next_node, ol, agent_dir = agent_id_dict[agent_id].prepare_agent(self.id)   
            go_vehs_list.append([agent_id, agent_next_node, go_link.id, ol, agent_dir])
        return go_vehs_list

    def non_conflict_vehs(self):
        self.go_vehs = []
        ### a primary direction
        in_links = [l for l in self.in_links.keys() if len(link_id_dict[l].queue_veh)>0]
        if len(in_links) == 0: return
        go_link = link_id_dict[random.choice(in_links)]
        go_vehs_list = self.find_go_vehs(go_link)
        self.go_vehs += go_vehs_list
        ### a non-conflicting direction
        if (np.min([veh[-1] for veh in go_vehs_list])<-45) or (go_link.type=='v'): return ### no opposite veh allows to move if there is left turn veh in the primary direction; or if the primary incoming link is a virtual link
        if self.in_links[go_link.id] == None: return ### no straight ahead opposite links
        op_go_link = link_id_dict[self.in_links[go_link.id]]
        op_go_link = link_id_dict[node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        op_go_vehs_list = self.find_go_vehs(op_go_link)
        self.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side

    def run_node_model(self, t_now, transfer_s, transfer_e):
        self.non_conflict_vehs()
        node_move = 0
        n_t_key_loc_flow = 0
        ### Agent reaching destination
        for [agent_id, next_node, il, ol, agent_dir] in self.go_vehs:
            ### arrival
            if (next_node is None) and (self.id == agent_id_dict[agent_id].destin_nid):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                n_t_key_loc_flow += agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'arr', il, ol, transfer_s, transfer_e)
            ### no storage capacity downstream
            elif link_id_dict[ol].st_c < 1:
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            elif (link_id_dict[il].ou_c >= 1) & (link_id_dict[ol].in_c >= 1):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                n_t_key_loc_flow += agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'flow', il, ol, transfer_s, transfer_e)
                link_id_dict[ol].receive_veh(agent_id)
            ### either inlink-sending or outlink-receiving or both exhaust
            else:
                control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]:
                    node_move += 1
                    ### before move agent as it uses the old agent.cl_enter_time
                    link_id_dict[il].send_veh(t_now, agent_id)
                    n_t_key_loc_flow += agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'chance', il, ol, transfer_s, transfer_e)
                    link_id_dict[ol].receive_veh(agent_id)
                else:
                    pass
        return node_move, n_t_key_loc_flow

class Link:
    def __init__(self, id, lanes, length, fft, capacity, type, start_nid, end_nid, geometry):
        ### input
        self.id = id
        self.lanes = lanes
        self.length = length
        self.fft = fft
        self.capacity = capacity
        self.type = type
        self.start_nid = start_nid
        self.end_nid = end_nid
        self.geometry = geometry
        ### derived
        self.store_cap = max(1, length*lanes/8)
        self.in_c = self.capacity/3600.0 # capacity in veh/s
        self.ou_c = self.capacity/3600.0
        self.st_c = self.store_cap # remaining storage capacity
        ### empty
        self.queue_veh = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.start_node = None
        self.end_node = None

    def send_veh(self, t_now, agent_id):
        self.queue_veh = [v for v in self.queue_veh if v!=agent_id]
        self.ou_c = max(0, self.ou_c-1)
        if self.type=='real': self.travel_time_list.append((t_now, t_now-agent_id_dict[agent_id].cl_enter_time))
    
    def receive_veh(self, agent_id):
        self.run_veh.append(agent_id)
        self.in_c = max(0, self.in_c-1)

    def run_link_model(self, t_now):
        for agent_id in self.run_veh:
            if agent_id_dict[agent_id].cl_enter_time < t_now - self.fft:
                self.queue_veh.append(agent_id)
        self.run_veh = [v for v in self.run_veh if v not in self.queue_veh]
        ### remaining spaces on link for the node model to move vehicles to this link
        self.st_c = self.store_cap - len(self.run_veh) - len(self.queue_veh) 
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600
    
    def update_travel_time(self, t_now, link_time_lookback_freq):
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            new_weight = np.mean([dur for (_, dur) in self.travel_time_list])
            g.update_edge(self.start_nid+1, self.end_nid+1, c_double(new_weight))

class Agent:
    def __init__(self, id, origin_nid, destin_nid, dept_time):
        #input
        self.id = id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
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

    def load_trips(self, t_now):
        if (self.dept_time == t_now):
            initial_edge = node2link_dict[self.route_igraph[0]]
            link_id_dict[initial_edge].run_veh.append(self.id)
            self.status = 'loaded'
            self.cl_enter_time = t_now

    def prepare_agent(self, node_id):
        assert self.cle == node_id, "agent next node {} is not the transferring node {}, route {}".format(self.cle, node_id, self.route_igraph)
        if self.destin_nid == node_id: ### current node is agent destination
            return None, None, 0 ### id, next_node, dir
        agent_next_node = [end for (start, end) in self.route_igraph if start == node_id][0]
        ol = node2link_dict[(node_id, agent_next_node)]
        x_start, y_start = node_id_dict[self.cls].lon, node_id_dict[self.cls].lat
        x_mid, y_mid = node_id_dict[node_id].lon, node_id_dict[node_id].lat
        x_end, y_end = node_id_dict[agent_next_node].lon, node_id_dict[agent_next_node].lat
        in_vec, out_vec = (x_mid-x_start, y_mid-y_start), (x_end-x_mid, y_end-y_mid)
        dot, det = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]), (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
        agent_dir = np.arctan2(det, dot)*180/np.pi 
        return agent_next_node, ol, agent_dir
    
    def move_agent(self, t_now, new_cls, new_cle, new_status, il, ol, transfer_s, transfer_e):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now
        #if self.id==349: print(t_now, self.cls, self.cle, self.status)
        ### pass key location
        if (il==transfer_s) and (ol==transfer_e): return 1
        else: return 0

    def get_path(self):
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

def network(network_file_edges=None, network_file_nodes=None, simulation_outputs=None, scen_nm=''):

    links_df0 = pd.read_csv(absolute_path+network_file_edges)
    ### tertiary and below
    links_df0['lanes'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 25, links_df0['maxmph'])
    ### primary
    links_df0['lanes'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 55, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 2000*links_df0['lanes']
    links_df0 = links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]

    nodes_df0 = pd.read_csv(absolute_path+network_file_nodes)

    ### Convert to mtx
    wgh = links_df0['fft']
    row = links_df0['start_igraph']
    col = links_df0['end_igraph']
    assert max(np.max(row)+1, np.max(col)+1) == nodes_df0.shape[0], 'nodes and links dimension do not match, row {}, col {}, nodes {}'.format(np.max(row), np.max(col), nodes_df0.shape[0])
    g_coo = ssparse.coo_matrix((wgh, (row, col)), shape=(nodes_df0.shape[0], nodes_df0.shape[0]))
    print(g_coo.shape, len(g_coo.data))
    sio.mmwrite(absolute_path+simulation_outputs+'/network_sparse.mtx', g_coo)
    # g_coo = sio.mmread(absolute_path+'/outputs/network_sparse.mtx'.format(folder))
    g = interface.readgraph(bytes(absolute_path+simulation_outputs+'/network_sparse.mtx', encoding='utf-8'))

    ### Create link and node objects
    nodes = []
    links = []
    for row in nodes_df0.itertuples():
        real_node = Node(getattr(row, 'node_id_igraph'), getattr(row, 'lon'), getattr(row, 'lat'), 'real', getattr(row, 'node_osmid'))
        virtual_node = real_node.create_virtual_node()
        virtual_link = real_node.create_virtual_link()
        nodes.append(real_node)
        nodes.append(virtual_node)
        links.append(virtual_link)
    for row in links_df0.itertuples():
        real_link = Link(getattr(row, 'edge_id_igraph'), getattr(row, 'lanes'), getattr(row, 'length'), getattr(row, 'fft'), getattr(row, 'capacity'), 'real', getattr(row, 'start_igraph'), getattr(row, 'end_igraph'), getattr(row, 'geometry'))
        links.append(real_link)

    return g, nodes, links

def demand(nodes_osmid_dict, phased_flag=False, demand_files=None):

    all_od_list = []
    for demand_file in demand_files:
        od = pd.read_csv(absolute_path + demand_file)
        
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])   
        if phased_flag == False: od['dept_time'] = 0
        else: od['dept_time'] = np.random.randint(low=0, high=3600*5, size=od.shape[0])
        
        od['origin_nid'] = od['origin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        od['destin_nid'] = od['destin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        all_od_list.append(od)

    all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
    all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
    # print('total numbers of agents from file ', all_od.shape)
    all_od = all_od.iloc[0:5000].copy()
    # print('total numbers of agents taken ', all_od.shape)

    agents = []
    for row in all_od.itertuples():
        agents.append(Agent(getattr(row, 'agent_id'), getattr(row, 'origin_nid'), getattr(row, 'destin_nid'), getattr(row, 'dept_time')))
        
    return agents

def map_sp(agent_id):
    
    subp_agent = agent_id_dict[agent_id]
    subp_agent.get_path()
    return (agent_id, subp_agent)

def route(scen_nm=''):
    
    ### Build a pool
    process_count = 10
    pool = Pool(processes=process_count)

    ### Find shortest pathes
    t_odsp_0 = time.time()
    map_agent = [k for k, v in agent_id_dict.items() if v.cle != None]
    res = pool.imap_unordered(map_sp, map_agent)

    ### Close the pool
    pool.close()
    pool.join()
    cannot_arrive = 0
    for (agent_id, subp_agent) in list(res):
        agent_id_dict[agent_id].find_route = subp_agent.find_route
        agent_id_dict[agent_id].route_igraph = subp_agent.route_igraph
        if subp_agent.find_route=='n_a': cannot_arrive += 1
    t_odsp_1 = time.time()

    if cannot_arrive>0: print('{} out of {} cannot arrive'.format(cannot_arrive, len(agent_id_dict)))
    return t_odsp_1-t_odsp_0, len(map_agent)

def main(random_seed=0, transfer_s=None, transfer_e=None, node_demand=None):
    random.seed(random_seed)
    np.random.seed(random_seed)
    global g, agent_id_dict, node_id_dict, link_id_dict, node2link_dict
    
    reroute_flag = True
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    phased_flag = False
    scen_nm = 'class_test'
    network_file_edges = '/projects/bolinas_stinson_beach/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/bolinas_stinson_beach/network_inputs/osm_nodes.csv'
    demand_files = ['/projects/bolinas_stinson_beach/demand_inputs/bolinas_od_{}_per_origin.csv'.format(node_demand)]
    simulation_outputs = '/projects/bolinas_stinson_beach/simulation_outputs'

    ### network
    g, nodes, links = network(
        network_file_edges = network_file_edges, network_file_nodes = network_file_nodes,
        simulation_outputs = simulation_outputs, scen_nm = scen_nm)
    # print('numbers of real/virtual links {}/{}, real/virtual nodes {}/{}'.format( 
        # len([1 for link in links if link.type=='real']), len([1 for link in links if link.type=='v']),
        # len([1 for node in nodes if node.type=='real']), len([1 for node in nodes if node.type=='v']) ))
    nodes_osmid_dict = {node.osmid: node.id for node in nodes if node.type=='real'}
    node2link_dict = {(link.start_nid, link.end_nid): link.id for link in links}
    link_id_dict = {link.id: link for link in links}
    node_id_dict = {node.id: node for node in nodes}
    for link_id, link in link_id_dict.items():
        node_id_dict[link.start_nid].out_links.append(link_id)
        node_id_dict[link.end_nid].in_links[link_id] = None
    for node_id, node in node_id_dict.items():
        node.calculate_straight_ahead_links()
    
    ### demand
    agents = demand(nodes_osmid_dict, 
        phased_flag = phased_flag, demand_files = demand_files)
    agent_id_dict = {agent.id: agent for agent in agents}
    print(type(agent_id_dict[0]))
    
    t_s, t_e = 0, 101
    move = 0
    route_time = 0
    route_cnts = 0
    key_loc_flow = []
    for t in range(t_s, t_e):
        t_key_loc_flow = 0
        ### routing
        if (t==0) or (reroute_flag) and (t%reroute_freq == 0):
            print(sys.getsizeof(agent_id_dict))
            ### update link travel time
            for link_id, link in link_id_dict.items(): link.update_travel_time(t, link_time_lookback_freq)
            ### route
            rt, rc = route(scen_nm=scen_nm)
            route_time += rt
            route_cnts += rc
        ### load agents
        for agent_id, agent in agent_id_dict.items(): agent.load_trips(t)
        ### link model
        for link_id, link in link_id_dict.items(): link.run_link_model(t)
        ### node model
        for node_id, node in node_id_dict.items(): 
            n_t_move, n_t_key_loc_flow = node.run_node_model(t, transfer_s, transfer_e)
            move += n_t_move
            t_key_loc_flow += n_t_key_loc_flow
        if t_key_loc_flow>0: key_loc_flow.append([t, t_key_loc_flow])
        if t%100==0: print(t, np.sum([1 for a in agent_id_dict.values() if a.status=='arr']), move, route_time, route_cnts)
    # print(key_loc_flow)
    # print(np.sum([1 for a in agent_id_dict.values() if a.status=='arr']))
    return key_loc_flow

if __name__ == ('__main__'):
    main(node_demand=3)