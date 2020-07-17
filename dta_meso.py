#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import time 
import json
import pyproj
import random
import logging 
import numpy as np
import pandas as pd 
import ctypes
import scipy.io as sio
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
import scipy.sparse as ssparse
from multiprocessing import Pool
from scipy.stats import truncnorm
### dir
absolute_path = '/home/bingyu/spatial_queue'
### user
sys.path.insert(0, absolute_path+'/../')
from sp import interface
import util.haversine as haversine

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
        self.in_links_lt = {} ### {in_link_id: straight_ahead_out_link_id, ...}
        self.out_links = []
        ### empty
        self.go_vehs = [] ### veh that moves in this time step
        self.status = None
        self.empty = 0
        self.left_turn = 0
        self.efficiency = 0

    def create_virtual_node(self):
        return Node('vn{}'.format(self.id), self.lon+0.001, self.lat+0.001, 'v')

    def create_virtual_link(self):
        return Link('n{}_vl'.format(self.id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.id), self.id, 'LINESTRING({} {}, {} {})'.format(self.lon+0.001, self.lat+0.001, self.lon, self.lat))
    
    def calculate_straight_ahead_links(self):
        for il in self.in_links.keys():
            (x_start, y_start) = link_id_dict[il].geometry.interpolate(0.9, normalized=True).coords[0]
            in_vec = (self.lon-x_start, self.lat-y_start)
            for ol in self.out_links:
                (x_end, y_end) = link_id_dict[ol].geometry.interpolate(0.1, normalized=True).coords[0]
                out_vec = (x_end-self.lon, y_end-self.lat)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                ol_dir = np.arctan2(det, dot)*180/np.pi
                if (abs(ol_dir)<=20) and link_id_dict[il].type=='real':
                    self.in_links[il].append(ol)
                if (ol_dir>70) and (ol_dir<110) and link_id_dict[il].type=='real':
                    self.in_links_lt[il].append(ol)

    def move_veh(self, agent_id, t_now, left_turn_allowed=True):
        ### veh properties
        veh = agent_id_dict[agent_id]
        veh_len = veh.veh_len
        agent_destin_idsp = veh.destin_idsp
        ### this link
        agent_cls = veh.cls
        il = node2link_dict[(agent_cls, self.id)]
        ### arrival, current node is destination
        if self.id+1 == agent_destin_idsp:
            ### before move agent as it uses the old agent.cl_enter_time
            link_id_dict[il].send_veh(t_now, agent_id)
            veh.move_agent(t_now, self.id, None, 'arr')
            # if agent_id == 1042: print(agent_id_dict[agent_id].cls)
            return 1, False

        ### next link
        next_link_end = [end for (start, end) in veh.route_igraph if start == self.id][-1]
        ol = node2link_dict[(self.id, next_link_end)]
        ### need reroute because of spillback or fire closure ahead
        if (link_id_dict[ol].st_c < veh_len) or (link_id_dict[ol].closure_status=='closed'):
            veh.find_route = 'trapped'
            if link_id_dict[ol].st_c < veh_len:
                veh.find_route += '_t'
            if link_id_dict[ol].closure_status=='closed':
                veh.find_route += '_f'
            ### detour only at some frequency
            if t_now%30 != 0:
                pass
            ### temporarily increase the weight of the undesired road
            else:
                temp_close = []
                if link_id_dict[il].type =='real':
                    temp_close.append([agent_cls, self.id, link_id_dict[il].travel_time])
                for one_ol in self.out_links:
                    if (link_id_dict[one_ol].st_c < veh_len) or (link_id_dict[one_ol].closure_status=='closed'):
                        temp_close.append([self.id, link_id_dict[one_ol].end_nid, link_id_dict[one_ol].travel_time])
                    elif link_id_dict[one_ol].end_nid == agent_cls:
                        temp_close.append([self.id, link_id_dict[one_ol].end_nid, link_id_dict[one_ol].travel_time])
                    else:
                        pass
                for [temp_close_s, temp_close_e, temp_close_t] in temp_close:
                    g.update_edge(temp_close_s+1, temp_close_e+1, ctypes.c_double(10e7))
                sp = g.dijkstra(self.id+1, agent_destin_idsp)
                dist = sp.distance(agent_destin_idsp)
                if dist >= 10e7: ### no alternative path, do not update existing path as the trap status may be lifted in the next time step
                    # agent_id_dict[agent_id].route_igraph = []
                    # agent_id_dict[agent_id].find_route = 'trapped'
                    pass
                else:
                    sp_route = sp.route(agent_destin_idsp)
                    veh.route_igraph = [(agent_cls, self.id)] + [(start_sp-1, end_sp-1) for (start_sp, end_sp) in sp_route]
                    veh.find_route = 'detour'
                    ### if not trapped, i.e., "detour" or "normal" or "a", recalculate path related variables
                    next_link_end = [end for (start, end) in veh.route_igraph if start == self.id][-1]
                    ol = node2link_dict[(self.id, next_link_end)]
                sp.clear()
                for [temp_close_s, temp_close_e, temp_close_t] in temp_close:
                    g.update_edge(temp_close_s+1, temp_close_e+1, ctypes.c_double(temp_close_t))
        else:
            veh.find_route = 'normal'
        
        ### if vehicle is trapped, it does not move and also blocks other vehicles
        if veh.find_route[0:7] == 'trapped':
            veh.status = 'stay'
            return 0, False

        ### check if the movement is a left turn
        left_turn = (ol in self.in_links_lt[il])
        ### no storage capacity downstream, or closed downstream due to fire should already be dealt with in the "trapped" scenario
        ### if no left turns are allowed and this vehicle needs a left turn, then it cannot move
        if left_turn and left_turn_allowed==False:
            veh.status = 'stay'
            return 0, left_turn
        ### inlink-sending, outlink-receiving both permits
        elif (link_id_dict[il].ou_c >= 1) & (link_id_dict[ol].in_c >= 1):
            ### before move agent as it uses the old agent.cl_enter_time
            link_id_dict[il].send_veh(t_now, agent_id)
            veh.move_agent(t_now, self.id, next_link_end, 'flow')
            link_id_dict[ol].receive_veh(agent_id)
            # if agent_id == 1042: print(agent_id_dict[agent_id].cls)
            return 1, left_turn
        ### either inlink-sending or outlink-receiving or both exhaust
        else:
            control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
            toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
            if toss_coin[0]:
                ### before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id)
                veh.move_agent(t_now, self.id, next_link_end, 'chance')
                link_id_dict[ol].receive_veh(agent_id)
                # if agent_id == 1042: print(agent_id_dict[agent_id].cls)
                return 1, left_turn
            else:
                veh.status = 'stay'
                if link_id_dict[il].ou_c < link_id_dict[ol].in_c:
                    link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c-1)
                elif link_id_dict[ol].in_c < link_id_dict[il].ou_c:
                    link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c-1)
                else:
                    link_id_dict[il].ou_c -= 1
                    link_id_dict[ol].in_c -= 1
                return 0, left_turn

    def run_node_model(self, t_now):
        go_link_move, op_go_link_move = 0, 0
        left_turn = False
        node_all_potential_move = 0

        ### all potential inlinks
        in_links = []
        for l in self.in_links.keys():
            l_obj = link_id_dict[l]
            if len(l_obj.queue_veh)>0:
                in_links.append(l)
                node_all_potential_move += len(l_obj.queue_veh)
            else:
                l_obj.empty += 1
        ### randomly select an inflow link
        if len(in_links) == 0:
            self.empty += 1
            return 0
        go_link = link_id_dict[random.choice(in_links)]
        ### non-blocking between different lanes of the same link
        go_vehs_list = go_link.queue_veh[0:int(np.floor(go_link.lanes))] ### if queue_len<lane, slice will automatically end at queue_len
        for veh in go_vehs_list:
            ### left turns are allowed for all vehicles from the primary direction
            veh_move, veh_left_turn = self.move_veh(veh, t_now, left_turn_allowed=True)
            go_link_move += veh_move
            self.left_turn += veh_left_turn
            left_turn = left_turn or veh_left_turn
        go_link.efficiency += go_link_move/len(go_vehs_list)
        ### if primary direction has left turns, then no secondary direction allowed
        if left_turn: 
            self.efficiency += go_link_move/node_all_potential_move
            return go_link_move
        
        ### get straight ahead direction as the secondary direction
        try:
            op_go_link = link_id_dict[random.choice(self.in_links[go_link.id])]
        except IndexError: ### cannot choose from emtpy sequence
            self.efficiency += go_link_move/node_all_potential_move
            return go_link_move
        try:
            op_go_link = link_id_dict[node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        except KeyError:
            self.efficiency += go_link_move/node_all_potential_move
            return go_link_move
        op_go_vehs_list = op_go_link.queue_veh[0:int(np.floor(op_go_link.lanes))]
        if len(op_go_vehs_list) == 0:
            self.efficiency += go_link_move/node_all_potential_move
            return go_link_move
        for veh in op_go_vehs_list:
            ### left turns not allowed for secondary direction if primary direction has vehicles moving
            if veh_move == 0:
                veh_move, veh_left_turn = self.move_veh(veh, t_now, left_turn_allowed=True)
            else:
                veh_move, veh_left_turn = self.move_veh(veh, t_now, left_turn_allowed=False)
            op_go_link_move += veh_move
            self.left_turn += veh_left_turn
        op_go_link.efficiency += op_go_link_move/len(op_go_vehs_list)
        self.efficiency += (go_link_move+op_go_link_move)/node_all_potential_move
        return go_link_move+op_go_link_move

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
        [self.start_lon_proj, self.end_lon_proj], [self.start_lat_proj, self.end_lat_proj] = pyproj.transform(
            pyproj.Proj('epsg:4326'), pyproj.Proj('epsg:26910'), 
            [self.geometry.coords[0][1], self.geometry.coords[-1][1]], [self.geometry.coords[0][0], self.geometry.coords[-1][0]])
        ### empty
        self.queue_veh = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.travel_time = fft
        self.queue_start = False
        self.queue_start_t = None
        self.queue_end_t = None
        self.closure_status = 'open'
        self.empty = 0 ### number of times when no vehicles queue
        self.efficiency = 0 ### cumsum([vehicles_leaving/max_possible_leaving])

    def get_closure_status(self, t_now, flame_length_hour):
        if flame_length_hour.shape[0]==0:
            return
        l13 = np.vstack((flame_length_hour['lon'] - self.start_lon_proj, flame_length_hour['lat'] - self.start_lat_proj)).T
        l23 = np.vstack((flame_length_hour['lon'] - self.end_lon_proj, flame_length_hour['lat'] - self.end_lat_proj)).T
        l12 = (self.end_lon_proj - self.start_lon_proj, self.end_lat_proj - self.start_lat_proj)
        l21 = (self.start_lon_proj - self.end_lon_proj, self.start_lat_proj - self.end_lat_proj)
        ### line distance
        line_dist = np.abs(np.matmul(l13, (l12[-1], -l12[0]))) / np.linalg.norm(l12)
        ### start_dist_array
        start_node_distance = np.linalg.norm(l13, axis=1)
        start_node_angle = np.matmul(l13, l12)
        ### end_dist_array
        end_node_distance = np.linalg.norm(l23, axis=1)
        end_node_angle = np.matmul(l23, l21)
        point_line_dist = np.where(start_node_angle<0, start_node_distance,
                                    np.where(end_node_angle<0, end_node_distance, line_dist))
        if np.min(point_line_dist) < 20: ### within 20=(1.414*30/2) m of the cell centroid. Cell width ~= 30m
            self.closure_status = 'closed'

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
        ### find queue duration
        if len(self.queue_veh)>self.lanes:
            self.queue_end_t = t_now
            if not self.queue_start:
                self.queue_start_t = t_now
                self.queue_start = True
    
    def update_travel_time(self, t_now, link_time_lookback_freq):
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            self.travel_time = np.mean([dur for (_, dur) in self.travel_time_list])
            g.update_edge(self.start_nid+1, self.end_nid+1, ctypes.c_double(self.travel_time))

class Agent:
    def __init__(self, id, origin_nid, destin_nid, dept_time, veh_len, a_type):
        #input
        self.id = id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
        self.veh_len = veh_len
        self.type = a_type ### local or visitor
        ### derived
        self.cls = 'vn{}'.format(self.origin_nid) # current link start node
        self.cle = self.origin_nid # current link end node
        self.origin_idsp = self.origin_nid + 1
        self.destin_idsp = self.destin_nid + 1
        ### Empty
        self.route_igraph = []
        self.find_route = None
        self.status = 'predepart'
        self.cl_enter_time = None
        self.avoid = [] ### avoid these roads

    def load_trips(self, t_now):
        if (self.dept_time == t_now):
            initial_edge = node2link_dict[self.route_igraph[0]]
            link_id_dict[initial_edge].run_veh.append(self.id)
            self.status = 'loaded'
            self.cl_enter_time = t_now
    
    def move_agent(self, t_now, new_cls, new_cle, new_status):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now

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

### distance to fire starting point
def fire_point_distance(veh_loc, fire_scen_id=1):
    if fire_scen_id == 1:
        fire_lon, fire_lat = -122.71454, 37.90623
    elif fire_scen_id == 2:
        fire_lon, fire_lat = -122.69970, 37.91324
    elif fire_scen_id == 3:
        fire_lon, fire_lat = -122.70242, 37.92365
    else:
        print('no such fire scen ID')
    [veh_lon, veh_lat] = zip(*veh_loc)
    veh_firestart_dist = haversine.haversine(np.array(veh_lat), np.array(veh_lon), fire_lat, fire_lon)
    return veh_firestart_dist
### distance to fire frontier
def fire_frontier_distance(fire_frontier, veh_loc, t):
    [veh_lon, veh_lat] = zip(*veh_loc)
    if t>=np.max(fire_frontier['t']):
        fire_frontier_now = fire_frontier.loc[fire_frontier['t'].idxmax(), 'geometry']
        veh_fire_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_now)
    else:
        t_before = np.max(fire_frontier.loc[fire_frontier['t']<=t, 't'])
        t_after = np.min(fire_frontier.loc[fire_frontier['t']>t, 't'])
        fire_frontier_before = fire_frontier.loc[fire_frontier['t']==t_before, 'geometry'].values[0]
        fire_frontier_after = fire_frontier.loc[fire_frontier['t']==t_after, 'geometry'].values[0]
        veh_fire_dist_before = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_before)
        veh_fire_dist_after = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_after)
        veh_fire_dist = veh_fire_dist_before * (t_after-t)/(t_after-t_before) + veh_fire_dist_after * (t-t_before)/(t_after-t_before)
    return veh_fire_dist

### numbers of vehicles that have left the evacuation zone / buffer distance
def outside_polygon(evacuation_zone, evacuation_buffer, veh_loc):
    [veh_lon, veh_lat] = zip(*veh_loc)
    evacuation_zone_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, evacuation_zone)
    evacuation_buffer_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)

def network(network_file_edges=None, network_file_nodes=None, simulation_outputs=None, cf_files=[], scen_nm=''):
    logger = logging.getLogger("bk_evac")

    links_df0 = pd.read_csv(absolute_path + network_file_edges)
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
            cf_link_df = pd.read_csv(absolute_path+ cf_file)
            cf_links.append(cf_link_df)
        cf_links_df = pd.concat(cf_links)
        ### along counterflow direction
        cf_links_id = cf_links_df.loc[cf_links_df['along']==1, 'edge_id_igraph']
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(cf_links_id), links_df0['lanes']*2, links_df0['lanes'])
        ### opposite counterflow direction
        opcf_links_id = cf_links_df.loc[cf_links_df['along']==0, 'edge_id_igraph']
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(opcf_links_id), 0, links_df0['lanes'])
        links_df0['maxmph'] = np.where(links_df0['edge_id_igraph'].isin(opcf_links_id), 0.000001, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 1900*links_df0['lanes']
    links_df0 = links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]

    nodes_df0 = pd.read_csv(absolute_path + network_file_nodes)

    ### Convert to mtx
    wgh = links_df0['fft']
    row = links_df0['start_igraph']
    col = links_df0['end_igraph']
    assert max(np.max(row)+1, np.max(col)+1) == nodes_df0.shape[0], 'nodes and links dimension do not match, row {}, col {}, nodes {}'.format(np.max(row), np.max(col), nodes_df0.shape[0])
    g_coo = ssparse.coo_matrix((wgh, (row, col)), shape=(nodes_df0.shape[0], nodes_df0.shape[0]))
    logger.info("({}, {}), {}".format(g_coo.shape[0], g_coo.shape[1], len(g_coo.data)))
    sio.mmwrite(absolute_path + simulation_outputs + '/network_sparse_{}.mtx'.format(scen_nm), g_coo)
    # g_coo = sio.mmread(absolute_path+'/outputs/network_sparse.mtx'.format(folder))
    g = interface.readgraph(bytes(absolute_path + simulation_outputs + '/network_sparse_{}.mtx'.format(scen_nm), encoding='utf-8'))

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

def demand(nodes_osmid_dict, precalculated=False, dept_time=None, demand_files=None, tow_pct=0, phase_scale=None):
    logger = logging.getLogger("bk_evac")

    all_od_list = []
    for demand_file in demand_files:
        od = pd.read_csv(absolute_path + demand_file)
        ### transform OSM based id to graph node id
        od['origin_nid'] = od['origin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        od['destin_nid'] = od['destin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        ### assign agent id
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
        ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
        if precalculated=='precalculated':
            pass
        else:
            [dept_time_mean, dept_time_std, dept_time_min, dept_time_max] = dept_time
            if (dept_time_std==0): 
                od['dept_time'] = dept_time_mean
            else:
                truncnorm_a, truncnorm_b = (dept_time_min-dept_time_mean)/dept_time_std, (dept_time_max-dept_time_mean)/dept_time_std
                od['dept_time'] = truncnorm.rvs(truncnorm_a, truncnorm_b, loc=dept_time_mean, scale=dept_time_std, size=od.shape[0])
                od['dept_time'] = od['dept_time'].astype(int)
        if phase_scale is not None:
            dist_unit, phase_min = phase_scale.split("-")
            dist_unit = int(dist_unit)
            phase_min = int(phase_min)
            od['o_lon'] = od['origin_nid'].apply(lambda x: node_id_dict[x].lon)
            od['o_lat'] = od['origin_nid'].apply(lambda x: node_id_dict[x].lat)
            od['fire_point_dist'] = fire_point_distance(zip(od['o_lon'], od['o_lat']))
            od['phase_offset'] = od['fire_point_dist'].apply(lambda x: min(25, max(0,(x-1000)//dist_unit))*phase_min*60) ### every 1000m from the fire start point, leave later phase_scale min
            od['dept_time'] += od['phase_offset']
        ### assign vehicle length
        od['veh_len'] = np.random.choice([8, 18], size=od.shape[0], p=[1-tow_pct, tow_pct])
        all_od_list.append(od)

    all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
    # all_od = all_od[all_od['type']=='local'].drop_duplicates(subset='parcel')
    all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
    logger.info('total numbers of agents from file {}'.format(all_od.shape))
    # all_od = all_od.iloc[0:3000].copy()
    logger.info('total numbers of agents taken {}'.format(all_od.shape))

    agents = []
    for row in all_od.itertuples():
        agents.append(Agent(getattr(row, 'agent_id'), getattr(row, 'origin_nid'), getattr(row, 'destin_nid'), getattr(row, 'dept_time'), getattr(row, 'veh_len'), getattr(row, 'type')))    
    return agents

def map_sp(agent_id):
    subp_agent = agent_id_dict[agent_id]
    subp_agent.get_path()
    return (agent_id, subp_agent)

def route(scen_nm=''):
    logger = logging.getLogger("bk_evac")
    
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

    if cannot_arrive>0: logger.info('{} out of {} cannot arrive'.format(cannot_arrive, len(agent_id_dict)))
    return t_odsp_1-t_odsp_0, len(map_agent)

def main(random_seed=None, fire_speed=None, dept_time_id=None, tow_pct=0, hh_veh=None, reroute_flag=None, phase_scale=None, counterflow=None, transfer_s=None, transfer_e=None, firescen=None, commscen=None, popscen=None, scen=''):
    ### logging and global variables
    random.seed(random_seed)
    np.random.seed(random_seed)
    dept_time_dict = {'imm': [0,0,0,1000], 'fst': [20*60,10*60,10*60,30*60], 'mid': [40*60,20*60,20*60,60*60], 'slw': [60*60,30*60,30*60,90*60], 'precalculated': None}
    dept_time = dept_time_dict[dept_time_id]
    global g, agent_id_dict, node_id_dict, link_id_dict, node2link_dict
    
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = '/projects/bolinas_stinson_beach/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/bolinas_stinson_beach/network_inputs/osm_nodes.csv'
    vphh = 2
    visitor = 300
    demand_files = ["/projects/bolinas_stinson_beach/demand_inputs/od/resident_visitor_od_rs{}_commscen{}_vphh{}_visitor{}.csv".format(random_seed, commscen, vphh, visitor)]
    simulation_outputs = '/projects/bolinas_stinson_beach/simulation_outputs'
    if counterflow=='beach':
        cf_files = ['/projects/bolinas_stinson_beach/network_inputs/contraflow_beach.csv']
    else:
        cf_files = []

    scen_nm = 'rs{}_f{}_c{}_vphh{}_vist{}{}'.format(random_seed, firescen, commscen, vphh, visitor, scen)
    logger = logging.getLogger("bk_evac")
    logging.basicConfig(filename=absolute_path+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logger.info(scen_nm)

    ### network
    g, nodes, links = network(
        network_file_edges = network_file_edges, network_file_nodes = network_file_nodes,
        simulation_outputs = simulation_outputs, cf_files = cf_files, scen_nm = scen_nm)
    # sp = g.dijkstra(37+1, 155+1)
    # sp_dist = sp.distance(155+1)
    # print(sp_dist)
    # sys.exit(0)
    nodes_osmid_dict = {node.osmid: node.id for node in nodes if node.type=='real'}
    node2link_dict = {(link.start_nid, link.end_nid): link.id for link in links}
    link_id_dict = {link.id: link for link in links}
    node_id_dict = {node.id: node for node in nodes}
    for link_id, link in link_id_dict.items():
        node_id_dict[link.start_nid].out_links.append(link_id)
        node_id_dict[link.end_nid].in_links[link_id] = []
        node_id_dict[link.end_nid].in_links_lt[link_id] = []
    for node in node_id_dict.values():
        node.calculate_straight_ahead_links()
    
    ### demand
    # evacuation_zone_df = pd.read_csv(open(absolute_path+'/projects/berkeley/demand_inputs/manual_evacuation_zone_shattuck.csv'))
    # evacuation_zone_gdf = gpd.GeoDataFrame(evacuation_zone_df, crs='epsg:4326', geometry=evacuation_zone_df['WKT'].map(loads))
    # evacuation_zone = evacuation_zone_gdf['geometry'].iloc[0]
    # evacuation_buffer = evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).to_crs('epsg:4326').iloc[0]

    agents = demand(nodes_osmid_dict, precalculated=dept_time_id, dept_time=dept_time, demand_files = demand_files, tow_pct=tow_pct, phase_scale=phase_scale)
    agent_id_dict = {agent.id: agent for agent in agents}

    ### fire growth
    fire_frontier = pd.read_csv(open(absolute_path + '/projects/bolinas_stinson_beach/demand_inputs/fire/point{}_10Hr_simplified.csv'.format(firescen)))
    fire_frontier['t'] = (fire_frontier['Hour']/100-9)*3600 ### suppose fire starts at 9am
    fire_frontier = gpd.GeoDataFrame(fire_frontier, crs='epsg:4326', geometry=fire_frontier['geometry'].map(loads))
    ### flame length
    flame_length = pd.read_csv(open(absolute_path + '/projects/bolinas_stinson_beach/demand_inputs/flame_length/point_firescen{}.csv'.format(firescen)))
    flame_length = gpd.GeoDataFrame(flame_length, crs='epsg:4326', geometry=[Point(xy) for xy in zip(flame_length.lon, flame_length.lat)]).to_crs('epsg:26910')
    flame_length['lon'], flame_length['lat'] = flame_length['geometry'].x, flame_length['geometry'].y
    
    t_s, t_e = 0, 7201
    sight = 0
    ### time step output
    with open(absolute_path + simulation_outputs+'/t_stats/t_stats_scen{}.csv'.format(scen_nm),'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'predepart', 'loaded', 'flow', 'chance', 'stay', 'arr', 'arr_local', 'arr_visitor', 'a', 'n_a', 'normal', 'detour', 'trapped_t', 'trapped_f', 'trapped_t_f', 'trapped_local', 'trapped_visitor', 'see_fire', 'move', 'avg_fdist', 'neg_fdist', 'closed_roads'])+"\n")

    for t in range(t_s, t_e):
        move = 0
        ### flame length
        if (t>0) and (t%3600 == 0):
            flame_length_hour = flame_length.loc[(flame_length['t_hour']==t/3600) & (flame_length['flame_length']>2)]
            print(t, flame_length_hour.shape)
        ### routing
        if (t==0) or (reroute_flag) and (t%reroute_freq == 0):
            ### update link travel time
            for link_id, link in link_id_dict.items(): link.update_travel_time(t, link_time_lookback_freq)
            ### route
            route(scen_nm=scen_nm)
        ### load agents
        # first bring forward departure time if sees fire
        undeparted_veh_id, undeparted_veh_loc = [], []
        for agent in agent_id_dict.values():
            if agent.status=='predepart':
                undeparted_veh_id.append(agent.id)
                undeparted_veh_loc.append((node_id_dict[agent.origin_nid].lon, node_id_dict[agent.origin_nid].lat))
        if len(undeparted_veh_id)==0: pass
        else:
            origin_fire_dist = fire_frontier_distance(fire_frontier, undeparted_veh_loc, t)
            see_fire_veh_id = np.array(undeparted_veh_id)[origin_fire_dist<400] ### sight distance
            sight += len(see_fire_veh_id)
            for a in see_fire_veh_id: agent_id_dict[a].dept_time = t
        # now load agents
        for agent in agent_id_dict.values(): agent.load_trips(t)
        ### link model
        for link_id, link in link_id_dict.items(): 
            # first updates closure status
            if (t>0) and (t%3600 == 0) and (link.type=='real'): link.get_closure_status(t, flame_length_hour)
            # then run link model
            link.run_link_model(t)
        ### node model
        for node in node_id_dict.values(): 
            n_t_move = node.run_node_model(t)
            move += n_t_move
        ### metrics
        if t%1 == 0:
            ### agent status
            predepart_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='predepart'])
            loaded_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='loaded'])
            flow_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='flow'])
            chance_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='chance'])
            stay_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='stay'])
            arrival_cnts = np.sum([1 for a in agent_id_dict.values() if a.status=='arr'])
            arrival_cnts_local = np.sum([1 for a in agent_id_dict.values() if (a.status=='arr') and (a.type=='local')])
            arrival_cnts_visitor = np.sum([1 for a in agent_id_dict.values() if (a.status=='arr') and (a.type=='visitor')])
            if arrival_cnts == len(agent_id_dict):
                logger.info("all agents arrive at destinations, {}".format(t))
                print(commscen, ' ', reroute_flag, ' ', "all agents arrive at destinations, {}".format(t))
                break
            ### find route status
            a_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='a'])
            n_a_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='n_a'])
            normal_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='normal'])
            detour_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='detour'])
            trapped_t_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='trapped_t'])
            trapped_f_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='trapped_f'])
            trapped_t_f_cnts = np.sum([1 for a in agent_id_dict.values() if a.find_route=='trapped_t_f'])
            trapped_local = np.sum([1 for a in agent_id_dict.values() if (a.find_route[0:7]=='trapped')  and (a.type=='local')])
            trapped_visitor = np.sum([1 for a in agent_id_dict.values() if (a.find_route[0:7]=='trapped')  and (a.type=='visitor')])
            ### danger level
            veh_loc = [link_id_dict[node2link_dict[(agent.cls, agent.cle)]].midpoint for agent in agent_id_dict.values() if agent.status != 'arr']
            veh_fire_dist = fire_frontier_distance(fire_frontier, veh_loc, t)
            avg_fire_dist, neg_dist = np.mean(veh_fire_dist), np.sum(veh_fire_dist<0)
            ### road closure
            closed_roads = np.sum([link.closure_status=='closed' for link in link_id_dict.values()])
            # outside_danger_cnts = np.sum(fire_point_distance(veh_loc)>5000)
            # outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
            with open(absolute_path + simulation_outputs+'/t_stats/t_stats_scen{}.csv'.format(scen_nm),'a') as t_stats_outfile:
                t_stats_outfile.write(",".join([str(x) for x in [t, predepart_cnts, loaded_cnts, flow_cnts, chance_cnts, stay_cnts, arrival_cnts, arrival_cnts_local, arrival_cnts_visitor, a_cnts, n_a_cnts, normal_cnts, detour_cnts, trapped_t_cnts, trapped_f_cnts, trapped_t_f_cnts, trapped_local, trapped_visitor, sight, move, round(avg_fire_dist,2), neg_dist, closed_roads]]) + "\n")

        ### stepped outputs
        if t%1200==0:
            link_output = pd.DataFrame([(link.id, len(link.queue_veh), len(link.run_veh), round(link.travel_time, 2), link.empty, round(link.efficiency, 2)) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'q', 'r', 't', 'empty', 'efficiency'])
            link_output.to_csv(absolute_path + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)

            node_output = pd.DataFrame([(node.id, node.empty, round(node.efficiency, 2), node.left_turn) for node in node_id_dict.values() if node.type=='real'], columns=['node_id', 'empty', 'efficiency', 'left_turn'])
            node_output.to_csv(absolute_path + simulation_outputs + '/node_stats/node_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
        # if (t<12700) and (t>12000):
        #     link_detailed_output = {link.id: {'queue': [(agent_id, agent_id_dict[agent_id].cl_enter_time) for agent_id in link.queue_veh], 'run': [(agent_id, agent_id_dict[agent_id].cl_enter_time) for agent_id in link.run_veh]} for link in link_id_dict.values()}
        #     with open(absolute_path+simulation_outputs+'/link_detailed_outputs/link_detail_{}_t{}.json'.format(scen_nm, t), 'w') as outfile:
        #         json.dump(link_detailed_output, outfile, indent=2)
        if t%100==0: 
            logger.info(",".join([str(x) for x in [t, ' || ', predepart_cnts, loaded_cnts, flow_cnts, chance_cnts, stay_cnts, arrival_cnts, arrival_cnts_local, arrival_cnts_visitor, ' || ', a_cnts, n_a_cnts, normal_cnts, detour_cnts, trapped_t_cnts, trapped_f_cnts, trapped_t_f_cnts, trapped_local, trapped_visitor, ' || ', sight, move, round(avg_fire_dist,2), neg_dist, closed_roads]]))
        if (t>0) and (t%3600 == 0): 
            logger.info("closed roads " + ", ".join([str(link.id) for link in link_id_dict.values() if link.closure_status=='closed']))
            print(scen_nm + ",".join([str(x) for x in [t, ' || ', predepart_cnts, loaded_cnts, flow_cnts, chance_cnts, stay_cnts, arrival_cnts, arrival_cnts_local, arrival_cnts_visitor, ' || ', a_cnts, n_a_cnts, normal_cnts, detour_cnts, trapped_t_cnts, trapped_f_cnts, trapped_t_f_cnts, trapped_local, trapped_visitor, ' || ', sight, move, round(avg_fire_dist,2), neg_dist, closed_roads]]))
    
    ### link queue duration
    pd.DataFrame([(link.id, link.queue_start_t, link.queue_end_t) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'queue_start_t', 'queue_end_t']).to_csv(absolute_path + simulation_outputs + '/link_stats/link_queue_duration_{}.csv'.format(scen_nm), index=False)

if __name__ == '__main__':
    # for commscen in [0, 1, 2]:
    #     for firescen in [1, 2, 3]:
    #         main(random_seed=0, dept_time_id='precalculated', firescen=firescen, commscen=commscen)
    # main(random_seed=0, dept_time_id='precalculated', firescen=3, commscen=0, popscen=1500)
    main(random_seed=0, counterflow='beach', dept_time_id='precalculated', firescen=3, commscen=0, popscen=1500, scen='_efficiency_cfb')


