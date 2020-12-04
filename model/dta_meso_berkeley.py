#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import time 
import random
import logging 
import numpy as np
import pandas as pd 
from ctypes import *
import scipy.io as sio
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
import scipy.sparse as ssparse
from multiprocessing import Pool
from scipy.stats import truncnorm
### dir
home_dir = '/home/bingyu/Documents/spatial_queue' # os.environ['HOME']+'/spatial_queue'
work_dir = '/home/bingyu/Documents/spatial_queue' # os.environ['WORK']+'/spatial_queue'
scratch_dir = '/home/bingyu/Documents/spatial_queue/projects/berkeley_trb/simulation_outputs' # os.environ['OUTPUT_FOLDER']
### user
sys.path.insert(0, home_dir+'/..')
from sp import interface
sys.path.insert(0, home_dir)
import util.haversine as haversine

class Node:
    def __init__(self, id, x, y, type, osmid=None):
        self.id = id
        self.x = x
        self.y = y
        self.type = type
        self.osmid = osmid
        ### derived
        # self.id_sp = self.id + 1
        self.in_links = {} ### {in_link_id: straight_ahead_out_link_id, ...}
        self.out_links = []
        self.go_vehs = [] ### veh that moves in this time step
        self.status = None

    def create_virtual_node(self):
        return Node('vn{}'.format(self.id), self.x+0.001, self.y+0.001, 'v')
        # return Node('vn{}'.format(self.id), self.x+1, self.y+1, 'v')

    def create_virtual_link(self):
        return Link('n{}_vl'.format(self.id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.id), self.id, loads('LINESTRING({} {}, {} {})'.format(self.x+0.001, self.y+0.001, self.x, self.y)))
        # return Link('n{}_vl'.format(self.id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.id), self.id, loads('LINESTRING({} {}, {} {})'.format(self.x+1, self.y+1, self.x, self.y)))
    
    def calculate_straight_ahead_links(self):
        for il in self.in_links.keys():
            x_start = node_id_dict[link_id_dict[il].start_nid].x
            y_start = node_id_dict[link_id_dict[il].start_nid].y
            in_vec = (self.x-x_start, self.y-y_start)
            sa_ol = None ### straight ahead out link
            ol_dir = 180
            for ol in self.out_links:
                x_end = node_id_dict[link_id_dict[ol].end_nid].x
                y_end = node_id_dict[link_id_dict[ol].end_nid].y
                out_vec = (x_end-self.x, y_end-self.y)
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

    def non_conflict_vehs(self, t_now):
        logger = logging.getLogger("bk_evac")
        global both_links, choose_spruce
        self.go_vehs = []
        ### a primary direction
        in_links = [l for l in self.in_links.keys() if len(link_id_dict[l].queue_veh)>0]
        # if self.id == 1626: print(t_now, in_links)
        
        if len(in_links) == 0: return
        ### roundabout
        roundabout_in_links = [3770, 37806, 20990, 20158, 6529, 20145, 5498, 20993, 6542]
        try:
            go_link = link_id_dict[[l for l in in_links if l in roundabout_in_links][0]]
        except IndexError:
            go_link = link_id_dict[random.choice(in_links)]
        # if self.id == 1626: print(t_now, go_link, self.in_links[go_link.id])
        if (28224 in in_links) and (19355 in in_links):
            both_links += 1
            if go_link.id == 28224: choose_spruce += 1
            # logging.info("\n" + str(t_now) + " " + " ".join([str(l) for l in in_links]) + " " + str(both_links) + " " + str(choose_spruce))
        go_vehs_list = self.find_go_vehs(go_link)
        self.go_vehs += go_vehs_list
        # if self.id == 1626: print(t_now, self.go_vehs[0])
        
        ### a non-conflicting direction
        if (min([veh[-1] for veh in go_vehs_list])<-45) or (go_link.type=='v'): return ### no opposite veh allows to move if there is left turn veh in the primary direction; or if the primary incoming link is a virtual link
        if self.in_links[go_link.id] == None: return ### no straight ahead opposite links
        op_go_link = link_id_dict[self.in_links[go_link.id]]
        try:
            op_go_link = link_id_dict[node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        except KeyError: ### straight ahead link is one-way
            return
        op_go_vehs_list = self.find_go_vehs(op_go_link)
        self.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side

    def run_node_model(self, t_now, transfer_s, transfer_e):
        logger = logging.getLogger("bk_evac")
        self.non_conflict_vehs(t_now=t_now)
        node_move = 0
        spruce_flow = 0
        hearst_flow = 0
        other_flow = 0 
        # if self.id == 1626: logging.info("all " + " ".join([str(item) for go_veh in self.go_vehs for item in go_veh]))
        ### Agent reaching destination
        for [agent_id, next_node, il, ol, agent_dir] in self.go_vehs:
            veh_len = agent_id_dict[agent_id].veh_len
            ### arrival
            if (next_node is None) and (self.id == agent_id_dict[agent_id].destin_nid):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'arr', il, ol, transfer_s, transfer_e)
                if self.id == 1626: ### hearst spruce intersection
                    if il == 28224: spruce_flow += 1
                    elif il == 19355: hearst_flow += 1
                    else: other_flow += 1
                    # logging.info(str(t_now) + str(agent_id) + str(next_node) + str(il) + str(ol) + str(agent_dir) + 'arr')
            ### no storage capacity downstream
            elif link_id_dict[ol].st_c < veh_len:
                # if self.id == 1626: ### hearst spruce intersection
                #     logging.info(str(il) + ' spillback')
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            elif (link_id_dict[il].ou_c >= 1) & (link_id_dict[ol].in_c >= 1):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'flow', il, ol, transfer_s, transfer_e)
                link_id_dict[ol].receive_veh(agent_id)
                if self.id == 1626: ### hearst spruce intersection
                    if il == 28224: spruce_flow += 1
                    elif il == 19355: hearst_flow += 1
                    else: other_flow += 1
                    # logging.info(str(t_now) + str(agent_id) + str(next_node) + str(il) + str(ol) + str(agent_dir) + 'flow')
            ### either inlink-sending or outlink-receiving or both exhaust
            else:
                control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]:
                    node_move += 1
                    ### before move agent as it uses the old agent.cl_enter_time
                    link_id_dict[il].send_veh(t_now, agent_id)
                    agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'chance', il, ol, transfer_s, transfer_e)
                    link_id_dict[ol].receive_veh(agent_id)
                    if self.id == 1626: ### hearst spruce intersection
                        if il == 28224: spruce_flow += 1
                        elif il == 19355: hearst_flow += 1
                        else: other_flow += 1
                        # logging.info(str(t_now) +" "+ str(agent_id) +" "+ str(next_node) +" "+ str(il) +" "+ str(ol) +" "+ str(agent_dir) +" "+ 'chance' +" "+ str(link_id_dict[il].ou_c) +" "+ str(link_id_dict[ol].in_c) +" "+ str(toss_coin[0]))
                else:
                    # if self.id == 1626: ### hearst spruce intersection
                        # logging.info("toss F " + str(t_now) +" "+ str(agent_id) +" "+ str(il) +" "+ str(ol) +" "+  str(link_id_dict[il].ou_c) +" "+ str(link_id_dict[ol].in_c) +" "+ str(toss_coin[0]) )
                    if link_id_dict[il].ou_c < link_id_dict[ol].in_c:
                       link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c-1)
                    elif link_id_dict[ol].in_c < link_id_dict[il].ou_c:
                        link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c-1)
                    else:
                        link_id_dict[il].ou_c -= 1
                        link_id_dict[ol].in_c -= 1
        return node_move, spruce_flow, hearst_flow, other_flow

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
        # self.geometry = loads(geometry)
        self.geometry = geometry
        ### derived
        self.store_cap = max(15, length*lanes) ### at least allow any vehicle to pass. i.e., the road won't block any vehicle because of the road length
        # self.store_cap = max(15, length*lanes, (self.capacity/3600+1)*15)
        self.in_c = self.capacity/3600.0 # capacity in veh/s
        self.ou_c = self.capacity/3600.0
        self.st_c = self.store_cap # remaining storage capacity
        self.midpoint = list(self.geometry.interpolate(0.5, normalized=True).coords)[0]
        ### empty
        self.queue_veh = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        self.shelter_veh = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.travel_time = fft
        self.start_node = None
        self.end_node = None
        self.gps_veh = 0
        self.nongps_veh = 0

    def send_veh(self, t_now, agent_id):
        ### remove the agent from queue
        self.queue_veh = [v for v in self.queue_veh if v!=agent_id]
        self.ou_c = max(0, self.ou_c-1)
        if self.type=='real':
            self.travel_time_list.append((t_now, t_now-agent_id_dict[agent_id].cl_enter_time))
            if agent_id_dict[agent_id].gps_reroute == 1: self.gps_veh += 1
            else: self.nongps_veh += 1
    
    def receive_veh(self, agent_id):
        self.run_veh.append(agent_id)
        self.in_c = max(0, self.in_c-1)

    def run_link_model(self, t_now):
        for agent_id in self.run_veh:
            if self.id in [8129, 13755]:
                print(self.id, self.run_veh, self.queue_veh)
                print(agent_id_dict[agent_id].route_igraph)
            if agent_id_dict[agent_id].cl_enter_time < t_now - self.fft:
                self.queue_veh.append(agent_id)
        self.run_veh = [v for v in self.run_veh if v not in self.queue_veh]
        ### remaining spaces on link for the node model to move vehicles to this link
        self.st_c = self.store_cap - sum([agent_id_dict[agent_id].veh_len for agent_id in self.run_veh+self.queue_veh])
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600
    
    def update_travel_time(self, t_now, link_time_lookback_freq):
        # if len(self.queue_veh) > 0:
        #     travel_time_tmp = [t_now - agent_id_dict[agent_id].cl_enter_time for agent_id in self.queue_veh]
        #     self.travel_time = sum(travel_time_tmp)/len(travel_time_tmp)
        #     g.update_edge(self.start_nid+1, self.end_nid+1, c_double(self.travel_time))
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            travel_time_tmp = [dur for (_, dur) in self.travel_time_list]
            self.travel_time = sum(travel_time_tmp)/len(travel_time_tmp)
            g.update_edge(self.start_nid+1, self.end_nid+1, c_double(self.travel_time))

class Agent:
    def __init__(self, id, origin_nid, destin_nid, dept_time, veh_len, gps_reroute):
        #input
        self.id = id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
        self.veh_len = veh_len
        self.gps_reroute = gps_reroute
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
        if (self.dept_time <= t_now) and (self.status is None) and (self.find_route=='a'):
            try:
                initial_edge = node2link_dict[self.route_igraph[0]]
            except IndexError:
                print(self.route_igraph)
                print(self.id, self.origin_nid, self.destin_nid, self.cle)
            link_id_dict[initial_edge].run_veh.append(self.id)
            self.status = 'loaded'
            self.cl_enter_time = t_now

    def prepare_agent(self, node_id):
        assert self.cle == node_id, "agent next node {} is not the transferring node {}, route {}".format(self.cle, node_id, self.route_igraph)
        if self.destin_nid == node_id: ### current node is agent destination
            return None, None, 0 ### id, next_node, dir
        agent_next_node = [end for (start, end) in self.route_igraph if start == node_id][0]
        ol = node2link_dict[(node_id, agent_next_node)]
        x_start, y_start = node_id_dict[self.cls].x, node_id_dict[self.cls].y
        x_mid, y_mid = node_id_dict[node_id].x, node_id_dict[node_id].y
        x_end, y_end = node_id_dict[agent_next_node].x, node_id_dict[agent_next_node].y
        in_vec, out_vec = (x_mid-x_start, y_mid-y_start), (x_end-x_mid, y_end-y_mid)
        dot, det = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]), (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
        agent_dir = np.arctan2(det, dot)*180/np.pi 
        return agent_next_node, ol, agent_dir
    
    def move_agent(self, t_now, new_cls, new_cle, new_status, il, ol, transfer_s, transfer_e):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now
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
            print(self.id)
        else:
            sp_route = sp.route(self.destin_idsp)
            self.route_igraph = [(self.cls, self.cle)] + [(start_sp-1, end_sp-1) for (start_sp, end_sp) in sp_route]
            self.find_route = 'a'
            sp.clear()

### distance to fire starting point
# def fire_point_distance(veh_loc):
#     fire_lon, fire_lat = -122.249261, 37.910399
#     [veh_lon, veh_lat] = zip(*veh_loc)
#     veh_firestart_dist = haversine.haversine(np.array(veh_lat), np.array(veh_lon), fire_lat, fire_lon)
#     return veh_firestart_dist
### distance to fire frontier
def fire_frontier_distance(fire_frontier, veh_loc, t):
    # [veh_lon, veh_lat] = zip(*veh_loc)
    vehicle_loc_array = np.array(veh_loc)
    if t>=np.max(fire_frontier['t']):
        fire_frontier_now = fire_frontier.loc[fire_frontier['t'].idxmax(), 'geometry']
        veh_fire_dist = haversine.point_to_vertex_dist(vehicle_loc_array, fire_frontier_now)
    else:
        t_before = np.max(fire_frontier.loc[fire_frontier['t']<=t, 't'])
        t_after = np.min(fire_frontier.loc[fire_frontier['t']>t, 't'])
        fire_frontier_before = fire_frontier.loc[fire_frontier['t']==t_before, 'geometry'].values[0]
        fire_frontier_after = fire_frontier.loc[fire_frontier['t']==t_after, 'geometry'].values[0]
        veh_fire_dist_before = haversine.point_to_vertex_dist(vehicle_loc_array, fire_frontier_before)
        veh_fire_dist_after = haversine.point_to_vertex_dist(vehicle_loc_array, fire_frontier_after)
        veh_fire_dist = veh_fire_dist_before * (t_after-t)/(t_after-t_before) + veh_fire_dist_after * (t-t_before)/(t_after-t_before)
        # print(veh_fire_dist_before, veh_fire_dist_after)
        # sys.exit(0)
    return veh_fire_dist# np.mean(veh_fire_dist), np.sum(veh_fire_dist<0)
### numbers of vehicles that have left the evacuation zone / buffer distance
def outside_polygon(evacuation_zone, evacuation_buffer, veh_loc):
    # [veh_lon, veh_lat] = zip(*veh_loc)
    vehicle_loc_array = np.array(veh_loc)
    evacuation_zone_dist = haversine.point_to_vertex_dist(vehicle_loc_array, evacuation_zone)
    evacuation_buffer_dist = haversine.point_to_vertex_dist(vehicle_loc_array, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)

def network(network_file_edges=None, network_file_nodes=None, simulation_outputs=None, cf_files=[], scen_nm=''):
    logger = logging.getLogger("bk_evac")

    links_df0 = pd.read_csv(work_dir + network_file_edges)
    links_df0 = gpd.GeoDataFrame(links_df0, crs='epsg:4326', geometry=links_df0['geometry'].map(loads))#.to_crs(3857)
    ### lane assumptions
    ### leave to OSM specified values for motorway and trunk
    links_df0['lanes'] = np.where(links_df0['type'].isin(['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link']), 2, links_df0['lanes'])
    links_df0['lanes'] = np.where(links_df0['type'].isin(['residential', 'unclassified']), 1, links_df0['lanes'])
    ### speed assumptions
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 55, links_df0['maxmph'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['secondary', 'secondary_link', 'tertiary', 'tertiary_link']), 25, links_df0['maxmph'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['residential', 'unclassified']), 25*0.8, links_df0['maxmph'])
    if len(cf_files)>0:
        ### read counterflow links
        cf_links = []
        for cf_file in cf_files:
            cf_link_df = pd.read_csv(work_dir + cf_file)
            cf_links.append(cf_link_df)
        cf_links_df = pd.concat(cf_links)
        ### along counterflow direction
        cf_links_id = cf_links_df.loc[cf_links_df['along']==1, 'edge_id_igraph']
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(cf_links_id), links_df0['lanes']*2, links_df0['lanes'])
        ### opposite counterflow direction
        opcf_links_id = cf_links_df.loc[cf_links_df['along']==0, 'edge_id_igraph']
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(opcf_links_id), 0, links_df0['lanes'])
        links_df0['maxmph'] = np.where(links_df0['edge_id_igraph'].isin(opcf_links_id), 0.0000001, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 1900*links_df0['lanes']
    links_df0 = links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]
    links_df0.to_csv(scratch_dir + simulation_outputs + '/modified_network_edges_{}.csv'.format(scen_nm), index=False)
    # sys.exit(0)

    nodes_df0 = pd.read_csv(work_dir + network_file_nodes)
    nodes_df0 = gpd.GeoDataFrame(nodes_df0, crs='epsg:4326', geometry=[Point(xy) for xy in zip(nodes_df0['lon'], nodes_df0['lat'])])#.to_crs(3857)
    nodes_df0['x'] = nodes_df0['geometry'].apply(lambda x: x.x)
    nodes_df0['y'] = nodes_df0['geometry'].apply(lambda x: x.y)

    ### Convert to mtx
    wgh = links_df0['fft']
    row = links_df0['start_igraph']
    col = links_df0['end_igraph']
    assert max(np.max(row)+1, np.max(col)+1) == nodes_df0.shape[0], 'nodes and links dimension do not match, row {}, col {}, nodes {}'.format(np.max(row), np.max(col), nodes_df0.shape[0])
    g_coo = ssparse.coo_matrix((wgh, (row, col)), shape=(nodes_df0.shape[0], nodes_df0.shape[0]))
    logging.info("({}, {}), {}".format(g_coo.shape[0], g_coo.shape[1], len(g_coo.data)))
    sio.mmwrite(scratch_dir + simulation_outputs + '/network_sparse_{}.mtx'.format(scen_nm), g_coo)
    # g_coo = sio.mmread(absolute_path+'/outputs/network_sparse.mtx'.format(folder))
    g = interface.readgraph(bytes(scratch_dir + simulation_outputs + '/network_sparse_{}.mtx'.format(scen_nm), encoding='utf-8'))

    ### Create link and node objects
    nodes = []
    links = []
    for row in nodes_df0.itertuples():
        real_node = Node(getattr(row, 'node_id_igraph'), getattr(row, 'x'), getattr(row, 'y'), 'real', getattr(row, 'node_osmid'))
        virtual_node = real_node.create_virtual_node()
        virtual_link = real_node.create_virtual_link()
        nodes.append(real_node)
        nodes.append(virtual_node)
        links.append(virtual_link)
    for row in links_df0.itertuples():
        real_link = Link(getattr(row, 'edge_id_igraph'), getattr(row, 'lanes'), getattr(row, 'length'), getattr(row, 'fft'), getattr(row, 'capacity'), 'real', getattr(row, 'start_igraph'), getattr(row, 'end_igraph'), getattr(row, 'geometry'))
        links.append(real_link)

    return g, nodes, links


def demand(nodes_osmid_dict, dept_time=[0,0,0,1000], demand_files=None, tow_pct=0, phase_tdiff=None, reroute_pct=0):
    logger = logging.getLogger("bk_evac")

    all_od_list = []
    [dept_time_mean, dept_time_std, dept_time_min, dept_time_max] = dept_time
    # demand_files = ['/projects/berkeley_trb/demand_inputs/od_test.csv']
    for demand_file in demand_files:
        od = pd.read_csv(work_dir + demand_file)
        ### transform OSM based id to graph node id
        od['origin_nid'] = od['o_osmid'].apply(lambda x: nodes_osmid_dict[x])
        od['destin_nid'] = od['d_osmid'].apply(lambda x: nodes_osmid_dict[x])
        ### assign agent id
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
        ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
        if (dept_time_std==0): od['dept_time'] = dept_time_mean
        else:
            truncnorm_a, truncnorm_b = (dept_time_min-dept_time_mean)/dept_time_std, (dept_time_max-dept_time_mean)/dept_time_std
            od['dept_time'] = truncnorm.rvs(truncnorm_a, truncnorm_b, loc=dept_time_mean, scale=dept_time_std, size=od.shape[0])
            od['dept_time'] = od['dept_time'].astype(int)
            # od.to_csv(scratch_dir + '/od.csv', index=False)
            # sys.exit(0)
        if phase_tdiff is not None:
            od['dept_time'] += (od['evac_zone']-1)*phase_tdiff*60 ### t_diff in minutes
        # od['dept_time'] = 0
        ### assign vehicle length
        od_tow = od.groupby(['apn', 'hh']).agg({'hh_veh': 'first'}).sample(frac=tow_pct).reset_index()
        od_tow['veh_len'] = 15
        od = od.merge(od_tow, how='left', on=['apn', 'hh', 'hh_veh']).fillna(value={'veh_len': 8})
        # print(od.drop_duplicates(subset=['apn']).shape, od.drop_duplicates(subset=['apn', 'hh']).shape, od.drop_duplicates(subset=['apn', 'hh', 'hh_veh']).shape, np.sum(od['veh_len']==8), np.sum(od['veh_len']==15))
        # sys.exit(0)
        ### assign rerouting choice
        od['gps_reroute'] = np.random.choice([0, 1], size=od.shape[0], p=[1-reroute_pct, reroute_pct])
        all_od_list.append(od)

    all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
    all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
    logging.info('total numbers of agents from file {}'.format(all_od.shape))
    # all_od = all_od.iloc[0:3000].copy()
    logging.info('total numbers of agents taken {}'.format(all_od.shape))

    agents = []
    for row in all_od.itertuples():
        agents.append(Agent(getattr(row, 'agent_id'), getattr(row, 'origin_nid'), getattr(row, 'destin_nid'), getattr(row, 'dept_time'), getattr(row, 'veh_len'), getattr(row, 'gps_reroute')))    
    return agents

def map_sp(agent_id):
    subp_agent = agent_id_dict[agent_id]
    subp_agent.get_path()
    return (agent_id, subp_agent)

def route(scen_nm='', who=None):
    logger = logging.getLogger("bk_evac")
    
    ### Build a pool
    process_count = 1 # int(os.environ['OMP_NUM_THREADS'])
    pool = Pool(processes=process_count)

    ### Find shortest pathes
    t_odsp_0 = time.time()
    if who=='all':
        map_agent = [k for k, v in agent_id_dict.items() if (v.cle != None) and (v.find_route != 'n_a')]
    elif who=='gps':
        map_agent = [k for k, v in agent_id_dict.items() if (v.cle != None) and (v.gps_reroute == True) and (v.find_route != 'n_a')]
    else:
        print('Unknown routing partition')
        sys.exit(0)
    res = pool.imap_unordered(map_sp, map_agent)

    ### Close the pool
    pool.close()
    pool.join()
    cannot_arrive = 0
    for (agent_id, subp_agent) in list(res):
        if subp_agent.find_route=='n_a':
            cannot_arrive += 1
            link_id_dict[node2link_dict[(subp_agent.cls, subp_agent.cle)]].shelter_veh.append(agent_id)
            link_id_dict[node2link_dict[(subp_agent.cls, subp_agent.cle)]].run_veh = [a for a in link_id_dict[node2link_dict[(subp_agent.cls, subp_agent.cle)]].run_veh if a != agent_id]
            link_id_dict[node2link_dict[(subp_agent.cls, subp_agent.cle)]].queue_veh = [a for a in link_id_dict[node2link_dict[(subp_agent.cls, subp_agent.cle)]].queue_veh if a != agent_id]
        agent_id_dict[agent_id].find_route = subp_agent.find_route
        agent_id_dict[agent_id].route_igraph = subp_agent.route_igraph
    t_odsp_1 = time.time()

    if cannot_arrive>0: logging.info('{} out of {} cannot arrive'.format(cannot_arrive, len(agent_id_dict)))
    return t_odsp_1-t_odsp_0, len(map_agent)


def main(random_seed=None, fire_speed=None, dept_time_id=None, tow_pct=None, hh_veh=None, reroute_pct=None, reroute_stp_id=None, phase_tdiff=None, counterflow=None, transfer_s=None, transfer_e=None):
    ### logging and global variables
    random.seed(random_seed)
    np.random.seed(random_seed)
    dept_time_dict = {'imm': [0,0,0,1000], 'fst': [20*60,10*60,10*60,30*60], 'mid': [40*60,20*60,20*60,60*60], 'slw': [60*60,30*60,30*60,90*60]}
    dept_time = dept_time_dict[dept_time_id]
    reroute_stp_dict = {'0': 3600*10, '1': 3600, '2': 7200, '0.1': 3600*0.1, '0.17': 600, '0.5': 3600*0.5}
    reroute_stp = reroute_stp_dict[reroute_stp_id]
    global g, agent_id_dict, node_id_dict, link_id_dict, node2link_dict

    global both_links, choose_spruce
    both_links = 0
    choose_spruce = 0
    
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = '/projects/berkeley_trb/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/berkeley_trb/network_inputs/osm_nodes.csv'
    demand_files = ["/projects/berkeley_trb/demand_inputs/od_rs{}_hhv{}.csv".format(random_seed, hh_veh)]
    simulation_outputs = '' ### scratch_folder
    if counterflow in [1]:
        cf_files = ['/projects/berkeley_trb/network_inputs/marin.csv', 
        '/projects/berkeley_trb/network_inputs/spruce.csv',
        '/projects/berkeley_trb/network_inputs/cedar.csv',
        '/projects/berkeley_trb/network_inputs/rose.csv',
        '/projects/berkeley_trb/network_inputs/shasta.csv',
        '/projects/berkeley_trb/network_inputs/oxfful.csv'
        ]
    elif counterflow in [2]:
        cf_files = ['/projects/berkeley_trb/network_inputs/marin.csv', 
        '/projects/berkeley_trb/network_inputs/spruce.csv',
        '/projects/berkeley_trb/network_inputs/cedar.csv',
        '/projects/berkeley_trb/network_inputs/rose.csv',
        '/projects/berkeley_trb/network_inputs/shasta.csv',
        '/projects/berkeley_trb/network_inputs/oxfful.csv',
        '/projects/berkeley_trb/network_inputs/shamlk.csv',
        '/projects/berkeley_trb/network_inputs/university.csv',
        ]
    else:
        cf_files = []

    scen_nm = 'link_gps_cnt_rs{}_f{}_dt{}_tow{}_hhv{}_r{}_rs{}_pt{}_cf{}'.format(random_seed, fire_speed, dept_time_id, tow_pct, hh_veh, reroute_pct, reroute_stp_id, phase_tdiff, counterflow)
    logger = logging.getLogger("bk_evac")
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(scen_nm)

    ### network
    g, nodes, links = network(
        network_file_edges = network_file_edges, network_file_nodes = network_file_nodes,
        simulation_outputs = simulation_outputs, cf_files = cf_files, scen_nm = scen_nm)
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
    evacuation_zone_df = pd.read_csv(open(work_dir+'/projects/berkeley_trb/demand_inputs/manual_evacuation_zone.csv'))
    evacuation_zone_gdf = gpd.GeoDataFrame(evacuation_zone_df, crs='epsg:4326', geometry=evacuation_zone_df['geometry'].map(loads))
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).to_crs('epsg:4326').unary_union
    # evacuation_buffer = evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).unary_union
    logging.info('Evacuation zone is {} km2, considering 1mile buffer it is {} km2'.format(evacuation_zone_gdf.to_crs('epsg:3857')['geometry'].unary_union.area/1e6, evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).unary_union.area/1e6))

    agents = demand(nodes_osmid_dict, dept_time=dept_time, demand_files = demand_files, tow_pct=tow_pct, phase_tdiff=phase_tdiff, reroute_pct=reroute_pct)
    agent_id_dict = {agent.id: agent for agent in agents}

    ### fire growth
    fire_frontier = pd.read_csv(open(work_dir + '/projects/berkeley_trb/demand_inputs/fire_fitted_ellipse.csv'))
    fire_frontier['t'] = (fire_frontier['t']-900)/fire_speed ### suppose fire starts at 11.15am
    fire_frontier = gpd.GeoDataFrame(fire_frontier, crs='epsg:4326', geometry=fire_frontier['geometry'].map(loads))#.to_crs(3857)
    
    t_s, t_e = 0, 3600*4+1
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'init', 'load', 'arr', 'move', 'avg_fdist', 'neg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")

    for t in range(t_s, t_e):
        move = 0
        ### routing
        if (t==0) or (reroute_pct>0) and (t%reroute_freq == 0) and (t<reroute_stp):
            ### update link travel time
            for link_id, link in link_id_dict.items(): link.update_travel_time(t, link_time_lookback_freq)
            ### route
            if t==0: route(scen_nm=scen_nm, who='all')
            else: route(scen_nm=scen_nm, who='gps')
        ### load agents
        for agent_id, agent in agent_id_dict.items(): agent.load_trips(t)
        ### link model
        for link_id, link in link_id_dict.items(): link.run_link_model(t)
        ### node model
        for node_id, node in node_id_dict.items(): 
            n_t_move, n_t_spruce_flow, n_t_hearst_flow, n_t_other_flow = node.run_node_model(t, transfer_s, transfer_e)
            move += n_t_move
        ### metrics
        if t%1 == 0:
            arrival_cnts = sum([1 for a in agent_id_dict.values() if a.status=='arr'])
            if arrival_cnts == len(agent_id_dict):
                logging.info("all agents arrive at destinations")
                break
            veh_loc, veh_loc_id = [], []
            for agent in agent_id_dict.values():
                if agent.status != 'arr':
                    veh_loc.append(link_id_dict[node2link_dict[(agent.cls, agent.cle)]].midpoint)
                    veh_loc_id.append(agent.id)
            veh_fire_dist = fire_frontier_distance(fire_frontier, veh_loc, t)
            # print(veh_loc[0:10])
            # print(fire_frontier['geometry'])
            # sys.exit(0)
            avg_fire_dist, neg_dist = np.mean(veh_fire_dist), np.sum(veh_fire_dist<0)
            for v_i in range(len(veh_loc_id)):
                if veh_fire_dist[v_i] <0:
                    agent_id_dict[veh_loc_id[v_i]].dept_time = t
            # outside_danger_cnts = np.sum(fire_point_distance(veh_loc)>5000)
            outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
            agent_init_cnt = sum([1 for agent in agent_id_dict.values() if agent.status is None])
            agent_load_cnt = sum([1 for agent in agent_id_dict.values() if agent.status=='loaded'])
            with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
                t_stats_outfile.write(",".join([str(x) for x in [t, agent_init_cnt, agent_load_cnt, arrival_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")
        ### stepped outputs
        if t%300==0:
            link_output = pd.DataFrame([(link.id, len(link.queue_veh), len(link.run_veh), len(link.shelter_veh), round(link.travel_time, 2)) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'q', 'r', 's', 't'])
            link_output[(link_output['q']>0) | (link_output['r']>0)].reset_index(drop=True).to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
            node_predepart = pd.DataFrame([(agent.cle, 1) for agent in agent_id_dict.values() if (agent.status in [None, 'loaded'])], columns=['node_id', 'predepart_cnt']).groupby('node_id').agg({'predepart_cnt': np.sum}).reset_index()
            node_predepart.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
        
        if t==t_e-10:
            link_gps_pct = pd.DataFrame([(link.id, link.gps_veh, link.nongps_veh) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'gps', 'non_gps'])
            link_gps_pct.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_gps_stats_{}.csv'.format(scen_nm), index=False)

        if t%100==0: 
            logging.info(" ".join([str(i) for i in [t, arrival_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + " " + str(len(veh_loc)))
        #     print(link_id_dict[20158].queue_veh, link_id_dict[20158].run_veh)

if __name__ == "__main__":
    main(random_seed=0, fire_speed=1, dept_time_id='mid', tow_pct=0.1, hh_veh='survey', reroute_pct=0.15, reroute_stp_id='0', phase_tdiff=0, counterflow=1)