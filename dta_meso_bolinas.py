#!/usr/bin/env python
# coding: utf-8

# In[22]:

import os
import gc
import sys
import time 
import random 
import numpy as np
import pandas as pd 
from ctypes import *
import geopy.distance
import scipy.io as sio
import geopandas as gpd
from shapely.wkt import loads
import scipy.sparse as ssparse
from operator import itemgetter
from scipy.stats import truncnorm
from shapely.geometry import Point
from multiprocessing import Pool, Process, Queue
from pympler.asizeof import asizeof

absolute_path = '/home/bingyu/spatial_queue'
sys.path.insert(0, absolute_path+'/../')
from sp import interface 
import util.haversine as haversine


# In[23]:


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
        try:
            op_go_link = link_id_dict[node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        except KeyError: ### straight ahead link is one-way
            return
        op_go_vehs_list = self.find_go_vehs(op_go_link)
        self.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side

    def run_node_model(self, t_now, transfer_s, transfer_e):
        self.non_conflict_vehs()
        node_move = 0
        n_t_key_loc_flow = 0
        ### Agent reaching destination
        for [agent_id, next_node, il, ol, agent_dir] in self.go_vehs:
            veh_len = agent_id_dict[agent_id].veh_len
            ### arrival
            if (next_node is None) and (self.id == agent_id_dict[agent_id].destin_nid):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                n_t_key_loc_flow += agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'arr', il, ol, transfer_s, transfer_e)
                if self.id == 1626: ### hearst spruce intersection
                    print(il, type(il))
            ### no storage capacity downstream
            elif link_id_dict[ol].st_c < veh_len:
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            elif (link_id_dict[il].ou_c >= 1) & (link_id_dict[ol].in_c >= 1):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                n_t_key_loc_flow += agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'flow', il, ol, transfer_s, transfer_e)
                link_id_dict[ol].receive_veh(agent_id)
                if self.id == 1626: ### hearst spruce intersection
                    print(il, type(il))
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
                    if self.id == 1626: ### hearst spruce intersection
                        print(il, type(il))
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
        self.geometry = loads(geometry)
        ### derived
        self.store_cap = max(18, length*lanes) ### at least allow any vehicle to pass. i.e., the road won't block any vehicle because of the road length
        self.in_c = self.capacity/3600.0 # capacity in veh/s
        self.ou_c = self.capacity/3600.0
        self.st_c = self.store_cap # remaining storage capacity
        self.midpoint = list(self.geometry.interpolate(0.5, normalized=True).coords)[0]
        ### empty
        self.queue_veh = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.travel_time = fft
        self.start_node = None
        self.end_node = None

    def send_veh(self, t_now, agent_id):
        ### remove the agent from queue
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
        self.st_c = self.store_cap - np.sum([agent_id_dict[agent_id].veh_len for agent_id in self.run_veh+self.queue_veh])
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600
    
    def update_travel_time(self, t_now, link_time_lookback_freq):
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            self.travel_time = np.mean([dur for (_, dur) in self.travel_time_list])
            g.update_edge(self.start_nid+1, self.end_nid+1, c_double(self.travel_time))

class Agent:
    def __init__(self, id, origin_nid, destin_nid, dept_time, veh_len):
        #input
        self.id = id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
        self.veh_len = veh_len
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


# In[24]:


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


# In[25]:


def demand(nodes_osmid_dict, dept_time=[0,0,0,1000], demand_files=None, tow_pct=0):

    all_od_list = []
    [dept_time_mean, dept_time_std, dept_time_min, dept_time_max] = dept_time
    for demand_file in demand_files:
        od = pd.read_csv(absolute_path + demand_file)
        ### assign agent id
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
        ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
        if (dept_time_std==0): od['dept_time'] = dept_time_mean
        else:
            truncnorm_a, truncnorm_b = (dept_time_min-dept_time_mean)/dept_time_std, (dept_time_max-dept_time_mean)/dept_time_std
            od['dept_time'] = truncnorm.rvs(truncnorm_a, truncnorm_b, loc=dept_time_mean, scale=dept_time_std, size=od.shape[0])
            od['dept_time'] = od['dept_time'].astype(int)
        ### assign vehicle length
        od['veh_len'] = np.random.choice([8, 18], size=od.shape[0], p=[1-tow_pct, tow_pct])
        ### transform OSM based id to graph node id
        od['origin_nid'] = od['origin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        od['destin_nid'] = od['destin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        all_od_list.append(od)

    all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
    all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
    print('total numbers of agents from file ', all_od.shape)
    # all_od = all_od.iloc[0:5000].copy()
    print('total numbers of agents taken ', all_od.shape)

    agents = []
    for row in all_od.itertuples():
        agents.append(Agent(getattr(row, 'agent_id'), getattr(row, 'origin_nid'), getattr(row, 'destin_nid'), getattr(row, 'dept_time'), getattr(row, 'veh_len')))
        
    return agents


# In[26]:


def map_sp(agent_id):
    
    subp_agent = agent_id_dict[agent_id]
    subp_agent.get_path()
    return (agent_id, subp_agent)

def route(scen_nm=''):
    
    ### Build a pool
    process_count = 50
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


def main(random_seed=0, reroute_flag=False, transfer_s=None, transfer_e=None, node_demand=None, dept_time=[0,0,0,1000], tow_pct=0, fire_speed=1, scen_nm='base'):
    random.seed(random_seed)
    np.random.seed(random_seed)
    global g, agent_id_dict, node_id_dict, link_id_dict, node2link_dict
    
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = '/projects/bolinas_stinson_beach/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/bolinas_stinson_beach/network_inputs/osm_nodes.csv'
    demand_files = ["/projects/bolinas_stinson_beach/demand_inputs/bolinas_od_3_per_origin.csv"]
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
    agents = demand(nodes_osmid_dict, dept_time=dept_time, demand_files = demand_files, tow_pct=tow_pct)
    agent_id_dict = {agent.id: agent for agent in agents}

    ### fire distance
    def outside_firezone(veh_loc, t):
        fire_lon, fire_lat = -122.22150, 37.86091
        [veh_lon, veh_lat] = zip(*veh_loc)
        veh_firestart_dist = haversine.haversine(np.array(veh_lat), np.array(veh_lon), fire_lat, fire_lon)
        return np.sum(veh_firestart_dist>5000)
    ### fire distance
    fire_frontier = pd.read_csv(open('projects/berkeley/demand_inputs/digitized_fire_frontier_by_hour_single_parts.csv'))
    fire_frontier['t'] = (fire_frontier['t']-900)/fire_speed ### suppose fire starts at 11.15am
    fire_frontier = gpd.GeoDataFrame(fire_frontier, crs={'init':'epsg:4326'}, geometry=fire_frontier['WKT'].map(loads))
    def fire_distance(veh_loc, t):
        t_before = np.max(fire_frontier.loc[fire_frontier['t']<=t, 't'])
        t_after = np.min(fire_frontier.loc[fire_frontier['t']>t, 't'])
        fire_frontier_before = fire_frontier.loc[fire_frontier['t']==t_before, 'geometry'].values.tolist()[0]
        fire_frontier_after = fire_frontier.loc[fire_frontier['t']==t_after, 'geometry'].values.tolist()[0]
        [veh_lon, veh_lat] = zip(*veh_loc)
        veh_fire_dist_before = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_before)
        veh_fire_dist_after = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_after)
        veh_fire_dist = veh_fire_dist_before * (t_after-t)/(t_after-t_before) + veh_fire_dist_after * (t-t_before)/(t_after-t_before)
        return np.mean(veh_fire_dist), np.sum(veh_fire_dist<0)
    
    t_s, t_e = 0, max(3600, dept_time[-1]+2000)
    move = 0
    t_stats = []
    for t in range(t_s, t_e):
    # for t in [0] + list(range(3749, 3753)):
        if (t==0): print(scen_nm)
        ### routing
        if (t==0) or (reroute_flag) and (t%reroute_freq == 0):
            ### update link travel time
            for link_id, link in link_id_dict.items(): link.update_travel_time(t, link_time_lookback_freq)
            ### route
            route(scen_nm=scen_nm)
        ### load agents
        for agent_id, agent in agent_id_dict.items(): agent.load_trips(t)
        ### link model
        for link_id, link in link_id_dict.items(): link.run_link_model(t)
        ### node model
        for node_id, node in node_id_dict.items(): 
            n_t_move, n_t_key_loc_flow = node.run_node_model(t, transfer_s, transfer_e)
            move += n_t_move
        ### metrics
        veh_loc = [link_id_dict[node2link_dict[(agent.cls, agent.cle)]].midpoint for agent in agent_id_dict.values() if agent.status != 'arr']
        avg_fire_dist, neg_dist = fire_distance(veh_loc, t)
        outside_danger_cnts = outside_firezone(veh_loc, t)
        arrival_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='arr'] )
        ### stepped outputs
        link_output = pd.DataFrame([(link.id, len(link.queue_veh), len(link.run_veh), round(link.travel_time, 2)) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'q', 'r', 't'])
        link_output[(link_output['q']>0) | (link_output['r']>0)].reset_index(drop=True).to_csv(absolute_path+simulation_outputs+'/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
        t_stats.append([t, arrival_cnts, move, round(avg_fire_dist,2), neg_dist, outside_danger_cnts])
        if t%100==0: print(t_stats[-1], len(veh_loc))
    
    pd.DataFrame(t_stats, columns=['t', 'arr', 'move', 'avg_fdist', 'neg_fdist', 'out_cnts']).to_csv(absolute_path+simulation_outputs+'/t_stats/t_stats_{}.csv'.format(scen_nm), index=False)

if __name__ == '__main__':

    dept_time_dict = {'imm': [0,0,0,1000], 'fst': [15*60,10*60,5*60,25*60], 'slw': [60*60,30*60,30*60,90*60]}
    main(reroute_flag=0, fire_speed=1, dept_time=dept_time_dict['imm'], tow_pct=0)
    # for fire_speed in [0.5, 1, 2]:
    #     for dept_time_id in ['imm', 'fst', 'slw']:
    #         for tow_pct in [0, 0.05, 0.1]:
    #             for reroute_flag in [0]:
    #                 scen_nm = 'r{}_f{}_dt{}_tow{}'.format(reroute_flag, fire_speed, dept_time_id, tow_pct)
    #                 if dept_time_id=='imm': continue
    #                 main(reroute_flag=reroute_flag, fire_speed=fire_speed, dept_time=dept_time_dict[dept_time_id], tow_pct=tow_pct, scen_nm=scen_nm)

    # %load_ext line_profiler
    # %lprun -f main main()
    main(reroute_flag=reroute_flag, fire_speed=fire_speed, dept_time=dept_time_dict[dept_time_id], tow_pct=tow_pct, scen_nm=scen_nm)





