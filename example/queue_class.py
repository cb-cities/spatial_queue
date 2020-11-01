#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time 
import random
import logging 
import numpy as np
import pandas as pd
import geopandas as gpd
from ctypes import c_double
import scipy.spatial.distance
from shapely.wkt import loads
from shapely.geometry import Point
# for coordinate transformation
from shapely.ops import transform
import pyproj
geo2prj = pyproj.Transformer.from_proj(pyproj.Proj('epsg:4326'), pyproj.Proj('epsg:3857'), always_xy=True)
### user
from sp import interface

class Network:
    def __init__(self):
        self.nodes = {}
        self.nodes_osmid_dict = {}
        self.links = {}
        self.node2link_dict = {}
        self.g = None
        self.agents = {}
        self.agents_stopped = {}
    
    def dataframe_to_network(self, network_file_edges=None, network_file_nodes=None):
        
        # nodes
        nodes_df = pd.read_csv(network_file_nodes)
        nodes_df = gpd.GeoDataFrame(nodes_df, crs='epsg:4326', geometry=[Point(x, y) for (x, y) in zip(nodes_df.lon, nodes_df.lat)]).to_crs('epsg:26910')
        nodes_df['x'] = nodes_df['geometry'].apply(lambda x: x.x)
        nodes_df['y'] = nodes_df['geometry'].apply(lambda x: x.y)
        # edges
        links_df = pd.read_csv(network_file_edges)
        links_df['fft'] = links_df['length']/links_df['maxmph']*2.237
        links_df['capacity'] = 1900*links_df['lanes']
        links_df = gpd.GeoDataFrame(links_df, crs='epsg:4326', geometry=links_df['geometry'].map(loads)).to_crs('epsg:26910')
        links_df = links_df[['eid', 'nid_s', 'nid_e', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]
        # links_df0.to_csv(scratch_dir + simulation_outputs + '/modified_network_edges.csv', index=False)
        
        ### Convert to mtx
        self.g = interface.from_dataframe(links_df, 'nid_s', 'nid_e', 'fft')

        ### Create link and node objects
        for row in nodes_df.itertuples():
            real_node = Node(getattr(row, 'nid'), getattr(row, 'x'), getattr(row, 'y'), 'real', getattr(row, 'osmid'))
            virtual_node = real_node.create_virtual_node()
            virtual_link = real_node.create_virtual_link()
            self.nodes[real_node.node_id] = real_node
            self.nodes_osmid_dict[real_node.osmid] = real_node.node_id
            self.nodes[virtual_node.node_id] = virtual_node
            self.links[virtual_link.link_id] = virtual_link
            self.node2link_dict[(virtual_link.start_nid, virtual_link.end_nid)] = virtual_link.link_id
        for row in links_df.itertuples():
            real_link = Link(getattr(row, 'eid'), getattr(row, 'lanes'), getattr(row, 'length'), getattr(row, 'fft'), getattr(row, 'capacity'), 'real', getattr(row, 'nid_s'), getattr(row, 'nid_e'), getattr(row, 'geometry'))
            self.links[real_link.link_id] = real_link
            self.node2link_dict[(real_link.start_nid, real_link.end_nid)] = real_link.link_id

    def add_connectivity(self):
        for link_id, link in self.links.items():
            self.nodes[link.start_nid].outgoing_links.append(link_id)
            self.nodes[link.end_nid].incoming_links[link_id] = dict()
        for node_id, node in self.nodes.items():
            for incoming_link in node.incoming_links.keys():
                for outgoing_link in node.outgoing_links:
                    # assigning 0 degree turn to all virtual incoming links
                    if isinstance(incoming_link, str) and (incoming_link[-2:] == 'vl'):
                        turning_angle = 0
                    else:
                        turning_angle = - self.links[incoming_link].angle + self.links[outgoing_link].angle
                    # adjust for more than 180 turning angles
                    if turning_angle >= np.pi:
                        turning_angle = 2*np.pi - turning_angle
                    if turning_angle <= -np.pi:
                        turning_angle = 2*np.pi + turning_angle
                    turning_angle = turning_angle/np.pi*180
                    node.incoming_links[incoming_link][outgoing_link] = turning_angle
    
    def add_demand(self, demand_files=None, dept_time_col=None, phase_tdiff=None, reroute_pct=0, tow_pct=0):
        all_od_list = []
        for demand_file in demand_files:
            od = pd.read_csv(demand_file)
            ### transform OSM based id to graph node id
            od['origin_nid'] = od['origin_osmid'].apply(lambda x: self.nodes_osmid_dict[x])
            od['destin_nid'] = od['destin_osmid'].apply(lambda x: self.nodes_osmid_dict[x])
            ### assign agent id
            if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
            ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
            od['dept_time'] = od[dept_time_col]
            ### assign vehicle length
            od['veh_len'] = np.random.choice([8, 18], size=od.shape[0], p=[1-tow_pct, tow_pct])
            ### assign rerouting choice
            od['gps_reroute'] = np.random.choice([0, 1], size=od.shape[0], p=[1-reroute_pct, reroute_pct])
            all_od_list.append(od)
        all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
        all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
        print('# agents from file ', all_od.shape[0])
        # all_od = all_od.iloc[0:3000].copy()
        for row in all_od.itertuples():
            self.agents[getattr(row, 'agent_id')] = Agent(getattr(row, 'agent_id'), getattr(row, 'origin_nid'), getattr(row, 'destin_nid'), getattr(row, 'dept_time'), getattr(row, 'veh_len'), getattr(row, 'gps_reroute')) 

class Node:
    def __init__(self, node_id, x, y, node_type, osmid=None):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type
        self.osmid = osmid
        ### derived
        self.all_queues = {}
        self.incoming_links = {} ### {in_link_id: straight_ahead_out_link_id, ...}
        self.outgoing_links = []
        self.go_vehs = [] ### veh that moves in this time step
        self.status = None

    def create_virtual_node(self):
        return Node('vn{}'.format(self.node_id), self.x+1, self.y+1, 'v')

    def create_virtual_link(self):
        return Link('n{}_vl'.format(self.node_id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.node_id), self.node_id, loads('LINESTRING({} {}, {} {})'.format(self.x+1, self.y+1, self.x, self.y)))
    
    def non_conflict_vehicles(self, link_id_dict=None, agent_id_dict=None, node2link_dict=None):
        non_conflict_go_vehicles = []
        left_turn_phase = False
        # all potential links
        incoming_links = [l for l in self.incoming_links.keys() if len(link_id_dict[l].queue_vehicles)>0]
        # no queuing vehicle at the intersection
        if len(incoming_links) == 0: return []
        # one primary direction
        go_link = link_id_dict[random.choice(incoming_links)]
        # vehicles from primary direction
        incoming_lanes = int(np.floor(go_link.lanes))
        incoming_vehs = len(go_link.queue_vehicles)
        for lane in range(min(incoming_lanes, incoming_vehs)):
            agent_id = go_link.queue_vehicles[lane]
            agent_incoming_link = go_link.link_id
            agent_outgoing_link = agent_id_dict[agent_id].next_link
            non_conflict_go_vehicles.append([agent_id, agent_incoming_link, agent_outgoing_link])
            if agent_outgoing_link is None: pass # current link end is destination/shelter
            else:
                try:
                    turning_angle = self.incoming_links[agent_incoming_link][agent_outgoing_link]
                except KeyError:
                    print(lane, go_link.link_id, go_link.queue_vehicles, agent_id)
                    print(agent_id_dict[agent_id].route, agent_incoming_link, agent_outgoing_link, self.incoming_links)
                    print(agent_id_dict[agent_id].current_link_start_nid, agent_id_dict[agent_id].current_link_end_nid, agent_id_dict[agent_id].next_link_end_nid)
                if turning_angle>=50: left_turn_phase=True
        # the chosen vehicle decides it is left turn phase
        if left_turn_phase: return non_conflict_go_vehicles

        # straight-ahead and right-turn phase
        try:
            go_link_reverse = node2link_dict[(go_link.end_nid, go_link.start_nid)]
        except KeyError:
            # primary direction is one-way
            return non_conflict_go_vehicles
        # find another incoming direction that aligns with the reverse link of the primary direction
        opposite_go_link_id = min(incoming_links, key = lambda x: np.abs( self.incoming_links[x][go_link_reverse] ))
        opposite_go_link = link_id_dict[opposite_go_link_id]
        incoming_lanes = int(np.floor(opposite_go_link.lanes))
        incoming_vehs = len(opposite_go_link.queue_vehicles)
        for lane in range(min(incoming_lanes, incoming_vehs)):
            agent_id = opposite_go_link.queue_vehicles[lane]
            agent_incoming_link = opposite_go_link.link_id
            agent_outgoing_link = agent_id_dict[agent_id].next_link
            # opposite direction will not have more left turns
            if agent_outgoing_link is None: turning_angle = 0
            else: turning_angle = self.incoming_links[agent_incoming_link][agent_outgoing_link]
            if turning_angle>=50: continue
            non_conflict_go_vehicles.append([agent_id, agent_incoming_link, agent_outgoing_link])
        return non_conflict_go_vehicles

    def run_node_model(self, t_now, node_id_dict=None, link_id_dict=None, agent_id_dict=None, node2link_dict=None, transfer_link_pairs=None):
        go_vehicles = self.non_conflict_vehicles( link_id_dict=link_id_dict, agent_id_dict=agent_id_dict, node2link_dict=node2link_dict)
        # if self.node_id==8205: print(go_vehicles, '\n\n')
        node_move = 0
        node_move_link_pairs = []
        ### hold subprocess results, avoid writing directly to global variable
        agent_update_dict = dict()
        link_update_dict = dict()
        all_incoming_links = set([v[1] for v in go_vehicles])
        all_outgoing_links = set([v[2] for v in go_vehicles])
        for il_id in all_incoming_links:
            il = link_id_dict[il_id]
            link_update_dict[il_id] = ['inflow', il.queue_vehicles, il.remaining_outflow_capacity, il.travel_time_list]
        for ol_id in all_outgoing_links:
            try:
                ol = link_id_dict[ol_id]
                link_update_dict[ol_id] = ['outflow', ol.run_vehicles, ol.remaining_inflow_capacity, ol.remaining_storage_capacity]
            except KeyError: ### ol_id is None
                pass

        for [agent_id, agent_il_id, agent_ol_id] in go_vehicles:
            # if agent_id==1627:
            #     # pass
            #     print('1627')
            agent = agent_id_dict[agent_id]
            ### link traversal time if the agent can pass
            travel_time = (t_now, t_now - agent.current_link_enter_time)
            ### track status of inflow link in the current iteration
            [_, inflow_link_queue_veh, inflow_link_ou_c, inflow_link_travel_time_list] = link_update_dict[agent_il_id]
            ### track status of outflow link in the current iteration
            try:
                [_, outflow_link_run_veh, outflow_link_in_c, outflow_link_st_c] = link_update_dict[agent_ol_id]
            except KeyError: # agent_ol_id is None
                pass

            ### arrival
            if self.node_id in [agent.destin_nid, agent.furthest_nid]:
                node_move += 1
                node_move_link_pairs.append((agent_il_id, agent_ol_id))
                ### before move agent as it uses the old agent.cl_enter_time
                link_update_dict[agent_il_id] = [
                    'inflow', [v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                if (agent.status=='shelter'):
                    agent_update_dict[agent_id] = ['shelter_arrive', self.node_id, None, t_now]
                else:
                    agent_update_dict[agent_id] = ['arrive', self.node_id, None, t_now]
            ### no storage capacity downstream
            elif outflow_link_st_c < agent.vehicle_length:
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            elif (inflow_link_ou_c >= 1) & (outflow_link_in_c >= 1):
                node_move += 1
                node_move_link_pairs.append((agent_il_id, agent_ol_id))
                ### before move agent as it uses the old agent.cl_enter_time
                link_update_dict[agent_il_id] = [ 
                    'inflow', [v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                link_update_dict[agent_ol_id] = [
                    'outflow', outflow_link_run_veh + [agent_id], max(0, outflow_link_in_c - 1), outflow_link_st_c-agent.vehicle_length]
                agent_update_dict[agent_id] = ['enroute', self.node_id, agent.next_link_end_nid, t_now]
            ### either inlink-sending or outlink-receiving or both exhaust
            else:
                control_cap = min(inflow_link_ou_c, outflow_link_in_c)
                toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]: ### vehicle can move
                    node_move += 1
                    node_move_link_pairs.append((agent_il_id, agent_ol_id))
                    ### before move agent as it uses the old agent.cl_enter_time
                    link_update_dict[agent_il_id] = [ 
                        'inflow', [v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                    link_update_dict[agent_ol_id] = [
                        'outflow', outflow_link_run_veh + [agent_id], max(0, outflow_link_in_c - 1), outflow_link_st_c-agent.vehicle_length]
                    agent_update_dict[agent_id] = ['enroute', self.node_id, agent.next_link_end_nid, t_now]
                else: ### even though vehicle cannot move, the remaining capacity needs to be adjusted
                    if inflow_link_ou_c < outflow_link_in_c: 
                        link_update_dict[agent_il_id] = [ 
                            'inflow', inflow_link_queue_veh, max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list]
                    elif inflow_link_ou_c > outflow_link_in_c: 
                        link_update_dict[agent_ol_id] = [
                            'outflow', outflow_link_run_veh, max(0, outflow_link_in_c - 1), outflow_link_st_c]
                    else:
                        link_update_dict[agent_il_id] = [ 
                            'inflow', inflow_link_queue_veh, max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list]
                        link_update_dict[agent_ol_id] = [
                            'outflow', outflow_link_run_veh, max(0, outflow_link_in_c - 1), outflow_link_st_c]
        return node_move, node_move_link_pairs, agent_update_dict, link_update_dict

class Link:
    def __init__(self, link_id, lanes, length, fft, capacity, link_type, start_nid, end_nid, geometry):
        ### input
        self.link_id = link_id
        self.lanes = lanes
        self.length = length
        self.fft = fft
        self.capacity = capacity
        self.link_type = link_type
        self.start_nid = start_nid
        self.end_nid = end_nid
        self.geometry = geometry
        ### derived
        self.storage_capacity = max(18, length*lanes) ### at least allow any vehicle to pass. i.e., the road won't block any vehicle because of the road length
        self.remaining_inflow_capacity = self.capacity/3600.0 # capacity in veh/s
        self.remaining_outflow_capacity = self.capacity/3600.0
        self.remaining_storage_capacity = self.storage_capacity # remaining storage capacity
        self.midpoint = list(self.geometry.interpolate(0.5, normalized=True).coords)[0]
        # angle
        # geometry_3857 = transform(geo2prj.transform, self.geometry)
        (node_1x, node_1y) = self.geometry.coords[0]
        node2 = self.geometry.interpolate(0.1, normalized=True)
        self.angle = np.arctan2(node2.y - node_1y, node2.x - node_1x)
        ### empty
        self.queue_vehicles = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_vehicles = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.travel_time = fft
        self.start_node = None
        self.end_node = None
        ### fire related
        self.status = 'open'
        self.burnt = None
        self.fire_time = None
        self.fire_type = None

    # @profile
    def run_link_model(self, t_now, agent_id_dict=None):
        for agent_id in self.run_vehicles:
            if agent_id_dict[agent_id].current_link_enter_time < t_now - self.fft:
                self.queue_vehicles.append(agent_id)
        self.run_vehicles = [agent_id for agent_id in self.run_vehicles if agent_id not in self.queue_vehicles]
        # if self.link_id == 'n8205_vl':
        #     print('link', self.start_nid, self.end_nid, self.run_vehicles, self.queue_vehicles, '\n')
        ### remaining spaces on link for the node model to move vehicles to this link
        self.remaining_storage_capacity = self.storage_capacity - sum([agent_id_dict[agent_id].vehicle_length for agent_id in self.run_vehicles+self.queue_vehicles])
        self.remaining_inflow_capacity, self.remaining_outflow_capacity = self.capacity/3600, self.capacity/3600
        if self.status=='closed': self.remaining_inflow_capacity = 0
    
    def update_travel_time(self, t_now, link_time_lookback_freq=None, g=None, update_graph=False):
        self.travel_time_list = [(t_rec, duration) for (t_rec, duration) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            self.travel_time = np.mean([dururation for (_, dururation) in self.travel_time_list])
            if update_graph: g.update_edge(self.start_nid, self.end_nid, c_double(self.travel_time))

    def close_link_to_newcomers(self, g=None):
        g.update_edge(self.start_nid, self.end_nid, c_double(1e8))
        self.status = 'closed'
        ### not updating fft and ou_c because current vehicles need to leave

class Agent:
    def __init__(self, agent_id, origin_nid, destin_nid, deptarture_time, vehicle_length, gps_reroute):
        #input
        self.agent_id = agent_id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.furthest_nid = destin_nid
        self.deptarture_time = deptarture_time
        self.vehicle_length = vehicle_length
        self.gps_reroute = gps_reroute
        ### derived
        self.current_link_start_nid = 'vn{}'.format(self.origin_nid) # current link start node
        self.current_link_end_nid = self.origin_nid # current link end node
        ### Empty
        self.route = {} ### None or list of route
        self.status = 'unloaded' ### unloaded, enroute, shelter, shelter_arrive, arrive
        self.current_link_enter_time = None
        self.next_link = None
        self.next_link_end_nid = None ### next link end

    def get_path(self, g=None):
        sp = g.dijkstra(self.current_link_end_nid, self.destin_nid)
        sp_dist = sp.distance(self.destin_nid)
        if sp_dist > 1e8:
            sp.clear()
            # self.route = {self.current_link_start_nid: self.current_link_end_nid}
            self.route = {}
            self.furthest_nid = self.current_link_end_nid
            self.status = 'shelter'
        else:
            sp_route = sp.route(self.destin_nid)
            ### create a path. Order only preserved in Python 3.7+. Do not rely on.
            # self.route = {self.current_link_start_nid: self.current_link_end_nid}
            for (start_nid, end_nid) in sp_route:
                self.route[start_nid] = end_nid
            sp.clear()

    def load_vehicle(self, t_now, node2link_dict=None, link_id_dict=None):
        ### for unloaded agents
        if (self.status=='unloaded'):
            if (self.deptarture_time == t_now):
                self.this_link = node2link_dict[ (self.current_link_start_nid, self.current_link_end_nid) ]
                link_id_dict[self.this_link].run_vehicles.append(self.agent_id)
                self.status = 'enroute'
                self.current_link_enter_time = t_now
                self.next_link_end_nid = self.route[self.current_link_end_nid]
                self.next_link = node2link_dict[ (self.current_link_end_nid, self.next_link_end_nid) ]

    def find_next_link(self, t_now, node2link_dict=None):
        ### for enroute vehicles
        # self.current_link_end_nid = self.route[self.current_link_start_nid]
        self.this_link = node2link_dict[ (self.current_link_start_nid, self.current_link_end_nid) ]
        try:
            self.next_link_end_nid = self.route[self.current_link_end_nid]
            self.next_link = node2link_dict[ (self.current_link_end_nid, self.next_link_end_nid) ]
        except KeyError:
            self.next_link_end_nid = None
            self.next_link = None
        