#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time 
import random
import logging 
import numpy as np
from ctypes import c_double
from shapely.wkt import loads
### dir
home_dir = '.' # os.environ['HOME']+'/spatial_queue'
### user
sys.path.insert(0, home_dir+'/..')
from sp import interface

class Node:
    def __init__(self, node_id, lon, lat, stype, osmid=None):
        self.id = node_id
        self.lon = lon
        self.lat = lat
        self.type = stype
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
    
    def calculate_straight_ahead_links(self, node_id_dict=None, link_id_dict=None):
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

    def find_go_vehs(self, go_link, agent_id_dict=None, node_id_dict=None, link_id_dict=None, node2link_dict=None):
        go_vehs_list = []
        incoming_lanes = int(np.floor(go_link.lanes))
        incoming_vehs = len(go_link.queue_veh)
        for ln in range(min(incoming_lanes, incoming_vehs)):
            agent_id = go_link.queue_veh[ln]
            try:
                agent_next_node, ol, agent_dir = agent_id_dict[agent_id].prepare_agent(self.id, node2link_dict=node2link_dict, node_id_dict=node_id_dict) 
            except AssertionError:
                print(agent_id, agent_id_dict[agent_id].status, agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle, self.id, self.in_links.keys(), go_link.id, go_link.queue_veh, link_id_dict[node2link_dict[(agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle)]].queue_veh)  
                # sys.exit(0)
            go_vehs_list.append([agent_id, agent_next_node, go_link.id, ol, agent_dir])
        return go_vehs_list

    def non_conflict_vehs(self, t_now, link_id_dict=None, agent_id_dict=None, node2link_dict=None, node_id_dict=None):
        
        # self.go_vehs = []
        go_vehs = []
        ### a primary direction
        in_links = [l for l in self.in_links.keys() if len(link_id_dict[l].queue_veh)>0]
        
        if len(in_links) == 0: return go_vehs
        go_link = link_id_dict[random.choice(in_links)]
        # if self.id == 1626: print(t_now, go_link, self.in_links[go_link.id])
        go_vehs_list = self.find_go_vehs(go_link, agent_id_dict=agent_id_dict, link_id_dict=link_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        # self.go_vehs += go_vehs_list
        go_vehs += go_vehs_list
        
        ### a non-conflicting direction
        if (np.min([veh[-1] for veh in go_vehs_list])<-45) or (go_link.type=='v'): return go_vehs ### no opposite veh allows to move if there is left turn veh in the primary direction; or if the primary incoming link is a virtual link
        if self.in_links[go_link.id] == None: return go_vehs ### no straight ahead opposite links
        op_go_link = link_id_dict[self.in_links[go_link.id]]
        try:
            op_go_link = link_id_dict[node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        except KeyError: ### straight ahead link is one-way
            return go_vehs
        op_go_vehs_list = self.find_go_vehs(op_go_link, agent_id_dict=agent_id_dict, link_id_dict=link_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        # self.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side
        go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45]
        return go_vehs

    def run_node_model(self, t_now, node_id_dict=None, link_id_dict=None, agent_id_dict=None, node2link_dict=None):
        go_vehs = self.non_conflict_vehs(t_now=t_now, link_id_dict=link_id_dict, agent_id_dict=agent_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        node_move = 0
        traffic_counter = {21806: 0, 11321:0}
        ### hold subprocess results, avoid writing directly to global variable
        agent_update_dict = dict()
        link_update_dict = dict()

        ### Agent reaching destination
        for [agent_id, next_node, il, ol, agent_dir] in go_vehs:
            agent = agent_id_dict[agent_id]
            veh_len = agent.veh_len
            ### link traversal time if the agent can pass
            if link_id_dict[il].type == 'real':
                travel_time = (t_now, t_now - agent.cl_enter_time)
            else:
                travel_time = None ### no update of travel time for virtual links
            ### track status of inflow link in the current iteration
            try:
                [inflow_link_queue_veh, inflow_link_ou_c, inflow_link_travel_time_list] = link_update_dict[il]
            except KeyError:
                inflow_link_queue_veh, inflow_link_ou_c, inflow_link_travel_time_list = link_id_dict[il].queue_veh, link_id_dict[il].ou_c, link_id_dict[il].travel_time_list
            ### track status of outflow link in the current iteration
            try:
                [outflow_link_run_veh, outflow_link_in_c] = link_update_dict[ol]
            except KeyError:
                if ol is None:
                    pass
                else:
                    outflow_link_run_veh, outflow_link_in_c= link_id_dict[ol].run_veh, link_id_dict[ol].in_c

            ### arrival
            if (agent.status=='shelter') or (self.id == agent.destin_nid):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                # link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict=agent_id_dict)
                # agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'arr', il, ol)
                link_update_dict[il] = [[v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                if (agent.status=='shelter'):
                    agent_update_dict[agent_id] = [self.id, next_node, 'shelter_arr', t_now]
                else:
                    agent_update_dict[agent_id] = [self.id, next_node, 'arr', t_now]
            ### no storage capacity downstream
            elif link_id_dict[ol].st_c < veh_len:
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            # elif (link_id_dict[il].ou_c >= 1) & (link_id_dict[ol].in_c >= 1):
            elif (inflow_link_ou_c >= 1) & (outflow_link_in_c >= 1):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                # link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict=agent_id_dict)
                # agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'flow', il, ol)
                # link_id_dict[ol].receive_veh(agent_id)
                link_update_dict[il] = [[v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                link_update_dict[ol] = [outflow_link_run_veh + [agent_id], max(0, outflow_link_in_c - 1)]
                agent_update_dict[agent_id] = [self.id, next_node, 'flow', t_now]
                # for count_loc in traffic_counter.keys():
                #     if ol == count_loc: traffic_counter[count_loc] += 1
            ### either inlink-sending or outlink-receiving or both exhaust
            else:
                # control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                control_cap = min(inflow_link_ou_c, outflow_link_in_c)
                toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]: ### vehicle can move
                    node_move += 1
                    ### before move agent as it uses the old agent.cl_enter_time
                    # link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict=agent_id_dict)
                    # agent_id_dict[agent_id].move_agent(t_now, self.id, next_node, 'chance', il, ol)
                    # link_id_dict[ol].receive_veh(agent_id)
                    link_update_dict[il] = [[v for v in inflow_link_queue_veh if v != agent_id], max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list+[travel_time]]
                    link_update_dict[ol] = [outflow_link_run_veh + [agent_id], max(0, outflow_link_in_c - 1)]
                    agent_update_dict[agent_id] = [self.id, next_node, 'chance', t_now]
                    # for count_loc in traffic_counter.keys():
                    #     if ol == count_loc: traffic_counter[count_loc] += 1
                else: ### even though vehicle cannot move, the remaining capacity needs to be adjusted
                    if inflow_link_ou_c < outflow_link_in_c: 
                        link_update_dict[il] = [inflow_link_queue_veh, max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list]
                    elif inflow_link_ou_c > outflow_link_in_c: 
                        link_update_dict[ol] = [outflow_link_run_veh, max(0, outflow_link_in_c - 1)]
                    else:
                        link_update_dict[il] = [inflow_link_queue_veh, max(0, inflow_link_ou_c - 1), inflow_link_travel_time_list]
                        link_update_dict[ol] = [outflow_link_run_veh, max(0, outflow_link_in_c - 1)]
                    # if link_id_dict[il].ou_c < link_id_dict[ol].in_c:
                    #     link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c-1)
                    # elif link_id_dict[ol].in_c < link_id_dict[il].ou_c:
                    #     link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c-1)
                    # else:
                    #     link_id_dict[il].ou_c -= 1
                    #     link_id_dict[ol].in_c -= 1
        return node_move, traffic_counter, agent_update_dict, link_update_dict

class Link:
    def __init__(self, link_id, lanes, length, fft, capacity, type, start_nid, end_nid, geometry):
        ### input
        self.id = link_id
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
        self.status = 'open'

    # def send_veh(self, t_now, agent_id, agent_id_dict=None):
    #     ### remove the agent from queue
    #     self.queue_veh = [v for v in self.queue_veh if v!=agent_id]
    #     self.ou_c = max(0, self.ou_c-1)
    #     if self.type=='real': self.travel_time_list.append((t_now, t_now - agent_id_dict[agent_id].cl_enter_time))
    
    # def receive_veh(self, agent_id):
    #     self.run_veh.append(agent_id)
    #     self.in_c = max(0, self.in_c-1)

    def run_link_model(self, t_now, agent_id_dict=None):
        for agent_id in self.run_veh:
            if agent_id_dict[agent_id].cl_enter_time < t_now - self.fft:
                self.queue_veh.append(agent_id)
        self.run_veh = [v for v in self.run_veh if v not in self.queue_veh]
        ### remaining spaces on link for the node model to move vehicles to this link
        self.st_c = self.store_cap - np.sum([agent_id_dict[agent_id].veh_len for agent_id in self.run_veh+self.queue_veh])
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600
        if self.status=='closed': self.in_c = 0
    
    def update_travel_time(self, t_now, link_time_lookback_freq=None, g=None, update_graph=False):
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            self.travel_time = np.mean([dur for (_, dur) in self.travel_time_list])
            if update_graph: g.update_edge(self.start_nid, self.end_nid, c_double(self.travel_time))

    def close_link_to_newcomers(self, g=None):
        g.update_edge(self.start_nid, self.end_nid, c_double(10e7))
        self.status = 'closed'
        ### not updating fft and ou_c because current vehicles need to leave

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
        ### Empty
        self.route_igraph = []
        self.find_route = None
        self.status = 'unloaded'
        self.arr_status = None
        self.cl_enter_time = None
        self.ol = None
        self.nle = None ### next link end

    def load_trips(self, t_now, node2link_dict=None, link_id_dict=None):
        if (self.status=='unloaded') & (self.dept_time == t_now):
            initial_edge = node2link_dict[self.route_igraph[0]]
            link_id_dict[initial_edge].run_veh.append(self.id)
            self.status = 'loaded'
            self.cl_enter_time = t_now

    def find_next_link(self, node2link_dict=None):
        
        if (self.status=='shelter') or (self.destin_nid == self.cle): ### current node is agent destination
            self.ol = None
        next_link = [(start, end) for (start, end) in self.route_igraph if start == self.cle]
        if next_link == []: ### before a route is assigned
            pass
        else:
            self.ol = node2link_dict[next_link[0]]
            self.nle = next_link[0][1]

    def prepare_agent(self, node_id, node2link_dict=None, node_id_dict=None):
        assert self.cle == node_id, "agent next node {} is not the transferring node {}, route {}".format(self.cle, node_id, self.route_igraph)
        if (self.status=='shelter') or (self.destin_nid == self.cle): ### current node is agent destination
            return None, None, 0
        x_start, y_start = node_id_dict[self.cls].lon, node_id_dict[self.cls].lat
        x_mid, y_mid = node_id_dict[node_id].lon, node_id_dict[node_id].lat
        x_end, y_end = node_id_dict[self.nle].lon, node_id_dict[self.nle].lat
        in_vec, out_vec = (x_mid-x_start, y_mid-y_start), (x_end-x_mid, y_end-y_mid)
        dot, det = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]), (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
        agent_dir = np.arctan2(det, dot)*180/np.pi 
        return self.nle, self.ol, agent_dir
    
    # def move_agent(self, t_now, new_cls, new_cle, new_status, il, ol):
    #     self.cls = new_cls
    #     self.cle = new_cle
    #     self.status = new_status
    #     self.cl_enter_time = t_now

    def get_path(self, g=None):
        sp = g.dijkstra(self.cle, self.destin_nid)
        sp_dist = sp.distance(self.destin_nid)

        if sp_dist > 10e7:
            sp.clear()
            return 'no_route'
        else:
            sp_route = sp.route(self.destin_nid)
            self.route_igraph = [(self.cls, self.cle)] + [(start_nid, end_nid) for (start_nid, end_nid) in sp_route]
            sp.clear()
            return 'find_route'

    def force_stop(self):
        # self.destin_nid = self.cle
        self.route_igraph = [(self.cls, self.cle)]
        self.status = 'shelter'
        self.ol = None
        self.nle = None