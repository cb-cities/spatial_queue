#!/usr/bin/env python
# coding: utf-8
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from random import shuffle
from ctypes import c_double
from shapely.wkt import loads
from shapely.ops import split
from shapely.geometry import Point
from shapely.geometry import LineString

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, abs_path)
sys.path.insert(0, "/home/bingyu/Documents")
from sp import interface

# import swifter
import multiprocessing

np.random.seed(0)
global movement_order

if __name__ == ('__main__'):
    import logging

def create_network(edges_f=None, nodes_f=None, special_nodes_f=None, ods_fs = None):

    nodes_df = pd.read_csv(nodes_f)
    edges_df = pd.read_csv(edges_f)
    ods_df = pd.concat([pd.read_csv(ods_f) for ods_f in ods_fs])

    ### nodes
    nodes_df['node_type'] = 'r'
    nodes_df = gpd.GeoDataFrame(nodes_df, crs='epsg:4326', geometry=[Point(xy) for xy in zip(nodes_df.lon, nodes_df.lat)]).to_crs('epsg:3857')
    nodes_df['x'] = nodes_df.geometry.x
    nodes_df['y'] = nodes_df.geometry.y
    nodes_df['osmid'] = nodes_df['osmid'].astype(str)

    ### edges
    edges_df['edge_type'] = 'r'
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads)).to_crs('epsg:3857')
    link_out_angle = dict()
    link_in_angle = dict()
    link_out_coord = dict()
    link_in_coord = dict()
    for link in edges_df.itertuples():
        link_id = getattr(link, 'eid')
        link_geometry = getattr(link, 'geometry')
        link_ss = link.geometry.interpolate(0.01, normalized=True)
        link_ee = link.geometry.interpolate(0.99, normalized=True)
        (link_sx, link_sy) = link_geometry.coords[0]
        (link_ex, link_ey) = link_geometry.coords[-1]
        link_ssx, link_ssy = link_ss.x, link_ss.y
        link_eex, link_eey = link_ee.x, link_ee.y
        link_in_angle[link_id] = np.arctan2(link_ssy-link_sy, link_ssx-link_sx)
        link_out_angle[link_id] = np.arctan2(link_ey-link_eey, link_ex-link_eex)
        link_in_coord[link_id] = [link_ssx + (-link_sy+link_ssy), link_ssy - (-link_sx+link_ssx)]
        link_out_coord[link_id] = [link_eex + (link_ey-link_eey), link_eey - (link_ex-link_eex)]

    ### demand

    ### virtual nodes and edges
    virtual_nodes_df = nodes_df.loc[nodes_df['osmid'].isin(np.unique(ods_df['origin_osmid'].astype(str)))].copy()
    node_evaczone_dict = {getattr(n, 'nid'): getattr(n, 'evac_zone') for n in nodes_df.itertuples()}
    virtual_nodes_df['evac_zone'] = virtual_nodes_df['nid'].map(node_evaczone_dict)
    virtual_nodes_df['nid'] = virtual_nodes_df['nid'].apply(lambda x: 'v{}'.format(x))
    virtual_nodes_df['node_type'] = 'v'
    virtual_nodes_df['x'] += 1
    virtual_nodes_df['y'] += 1
    nodes_df = pd.concat([nodes_df, virtual_nodes_df])

    virtual_edges_df = virtual_nodes_df[['nid', 'x', 'y']].copy()
    virtual_edges_df['nid_s'] = virtual_edges_df['nid']
    virtual_edges_df['nid_e'] = virtual_edges_df['nid_s'].apply(lambda x: int(x.replace('v', ''))) ### remove the 'v' from node id
    virtual_edges_df['eid'] = edges_df['eid'].shape[0] + np.arange(virtual_edges_df.shape[0])
    virtual_edges_df['edge_type'] = 'v'
    virtual_edges_df['length'] = 1
    virtual_edges_df['maxmph'] = 1000
    virtual_edges_df['lanes'] = 1e5
    virtual_edges_df['evac_zone'] = virtual_edges_df['nid_e'].map(node_evaczone_dict)
    virtual_edges_df['geometry'] = virtual_edges_df.apply(lambda e: LineString([(e['x'], e['y']), (e['x']-1, e['y']-1)]), axis=1)
    edge_columns = ['eid', 'nid_s', 'nid_e', 'edge_type', 'length', 'maxmph', 'lanes', 'evac_zone', 'geometry']
    edges_df = pd.concat([edges_df[edge_columns], virtual_edges_df[edge_columns]])
    # print(edges_df.head())

    edges_df['fft'] = edges_df['length']/edges_df['maxmph']*2.237
    edges_df['travel_time'] = edges_df['fft']
    edges_df['in_capacity'] = edges_df['lanes'] * 2000/3600
    edges_df['out_capacity'] = edges_df['lanes'] * 2000/3600
    edges_df['storage'] = edges_df['length'] * edges_df['lanes']
    edges_df['enter'] = 0
    g = interface.from_dataframe(edges_df[edges_df['edge_type']=='r'], 'nid_s', 'nid_e', 'fft')
    edge_nid_dict = {(getattr(edge, 'nid_s'), getattr(edge, 'nid_e')): getattr(edge, 'eid') for edge in edges_df.itertuples()}
    edge_fft_dict = {getattr(edge, 'eid'): getattr(edge, 'fft') for edge in edges_df.itertuples()}
    # edge_lanes_dict = {getattr(edge, 'eid'): getattr(edge, 'lanes') for edge in edges_df.itertuples()}
    # edge_capacity_dict = {getattr(edge, 'eid'): getattr(edge, 'in_capacity') for edge in edges_df.itertuples()}

    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def intersect(A, B, C, D):
        ### returns true if line segments AB and CD intersect
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    def intersection_movements():
        
        incoming_links = {n: [] for n in nodes_df['nid'].values.tolist()}
        outgoing_links = {n: [] for n in nodes_df['nid'].values.tolist()}
        nodes_movements = dict()
        movement_order = dict()

        for link in edges_df.itertuples():
            link_id = getattr(link, 'eid')
            incoming_links[getattr(link, 'nid_e')].append(link_id)
            outgoing_links[getattr(link, 'nid_s')].append(link_id)
        
        for node in nodes_df.itertuples():
            node_id = getattr(node, 'nid')
            node_movements_info = dict()
            for il in incoming_links[node_id]:
                for ol in outgoing_links[node_id]:
                    try:
                        ila, ola, ilc, olc = link_out_angle[il], link_in_angle[ol], link_out_coord[il], link_in_coord[ol]
                        node_movements_info[(il, ol)] = ila, ola, ilc, olc
                        movement_order[(il, ol)] = 1
                        if ila < 0: ila += 360
                        if ola < 0: ola += 360
                        if (ola-ila > 45) and (ola-ila < 135): movement_order[(il, ol)] = 2
                        if (ila-ola > 135) and (ila-ola < 315): movement_order[(il, ol)] = 2
                    except KeyError:
                        node_movements_info[(il, ol)] = 'v', 'v', 'v', 'v'
                        movement_order[(il, ol)] = 3
            for (il, ol), [ila, ola, ilc, olc] in node_movements_info.items():
                # nodes_movements['{}-{}'.format(il, ol)] = dict()
                for (il_, ol_), [ila_, ola_, ilc_, olc_] in node_movements_info.items():
                    if (il==il_) and (ol==ol_):
                        continue
                    elif (ilc=='v') or (ilc_=='v'):
                        ### virtual links
                        nodes_movements[(il, ol, il_, ol_)] = 1
                    elif intersect(ilc, olc, ilc_, olc_):
                        continue
                    else:
                        nodes_movements[(il, ol, il_, ol_)] = 1
        return nodes_movements, movement_order

    nodes_movements, movement_order = intersection_movements()

    return edges_df, nodes_df, ods_df, g, edge_nid_dict, edge_fft_dict, nodes_movements, movement_order

def create_agents(ods_df, nodes_df, edge_nid_dict):

    osmid2nid_dict = {getattr(node, 'osmid'): getattr(node, 'nid') for node in nodes_df[nodes_df['node_type']!='v'].itertuples()}

    agents_df = ods_df
    agents_df['agent_id'] = np.arange(agents_df.shape[0])
    agents_df['origin_nid'] = agents_df['origin_osmid'].astype(str).map(osmid2nid_dict)
    agents_df['destin_osmid'] = 'vsink_chico'
    agents_df['destin_nid'] = agents_df['destin_osmid'].map(osmid2nid_dict)
    agents_df['cl_enid'] = agents_df['origin_nid']
    agents_df['cl_snid'] = agents_df['origin_nid'].apply(lambda x: 'v{}'.format(x))
    agents_df['current_link'] = agents_df.set_index(['cl_snid', 'cl_enid']).index.map(edge_nid_dict)
    agents_df['cl_fft'] = np.nan
    agents_df['next_link'] = np.nan
    agents_df['onlink_status'] = 0 ### 0: empty; 1: run; 2: queue
    agents_df['agent_status'] = 0 ### 0: unloaded; 1: enroute; 2: shelter; -1: arrive
    agents_df['current_link_enter_time'] = np.nan
    agents_df = agents_df[['agent_id', 'origin_nid', 'destin_nid', 'evac_zone', 'agent_status', 'onlink_status', 'current_link', 'current_link_enter_time', 'cl_fft', 'cl_enid', 'next_link']]

    return agents_df

# @profile
def load_agents(agents_df, agent_routes, edge_nid_dict, t):
    
    ### load agent
    load_ids = t == agents_df['departure_time']
    if np.sum(load_ids) == 0:
        return agents_df

    agents_df.loc[load_ids, 'cl_fft'] = 0
    agents_df.loc[load_ids, 'nl_enid'] = agents_df.loc[load_ids].apply(lambda a: agent_routes[getattr(a, 'agent_id')][getattr(a, 'cl_enid')], axis=1)
    agents_df.loc[load_ids, 'next_link'] = agents_df.loc[load_ids].set_index(['cl_enid', 'nl_enid']).index.map(edge_nid_dict)
    agents_df['onlink_status'] = np.where(load_ids, 1, agents_df['onlink_status'])
    agents_df['agent_status'] = np.where(load_ids, 1, agents_df['agent_status'])
    agents_df['current_link_enter_time'] = np.where(load_ids, t, agents_df['current_link_enter_time'])
    
    return agents_df

# @profile
def link_update(agents_df, edges_df, g, t):

    link_undeparted_dict = agents_df.loc[agents_df['onlink_status']==0].groupby('current_link').size().to_dict()
    link_queue_dict = agents_df.loc[agents_df['onlink_status']==2].groupby('current_link').size().to_dict()
    link_run_dict = agents_df.loc[agents_df['onlink_status']==1].groupby('current_link').size().to_dict()
    link_enter_dict = agents_df.loc[agents_df['current_link_enter_time']==(t-1)].groupby('current_link').size().to_dict()
    edges_df['undeparted'] = edges_df['eid'].map(link_undeparted_dict).fillna(0)
    edges_df['queue'] = edges_df['eid'].map(link_queue_dict).fillna(0)
    edges_df['run'] = edges_df['eid'].map(link_run_dict).fillna(0)
    edges_df['storage_remain'] = edges_df['storage'] - (edges_df['run'] + edges_df['queue'])*8
    edges_df['enter'] += edges_df['eid'].map(link_enter_dict).fillna(0)
    
    # if t%300 == 0:
    #     new_travel_time = edges_df['queue']/ (edges_df['out_capacity'] + 0.1) + edges_df['fft']
    #     edges_to_update = np.abs(edges_df['travel_time']-new_travel_time)>10
    #     edges_df.loc[edges_to_update, 'travel_time'] = new_travel_time[edges_to_update] 
    #     for e in edges_df.loc[edges_to_update].itertuples():
    #         g.update_edge(getattr(e, 'nid_s'), getattr(e, 'nid_e'), c_double(getattr(e, 'travel_time')))
    
    return edges_df, g 

# def f(k):
#     return movement_order.get(k)

# @profile
def node_model(agents_df, edges_df, agent_routes, edge_nid_dict, edge_fft_dict, no_conflict_movement, movement_order, t):
    
    ### queue vehicles
    agents_df['onlink_status'] = np.where((agents_df['agent_status']==1) & (agents_df['cl_fft']<(t-agents_df['current_link_enter_time'])), 2, agents_df['onlink_status'])
    queue_agents_df = agents_df[agents_df['onlink_status']==2].copy().reset_index(drop=True)
    if queue_agents_df.shape[0] == 0:
        return agents_df
    
    # with multiprocessing.Pool(processes=4) as pool:
    #     res = pool.map(f, queue_agents_df[['current_link', 'next_link']].to_records(index=False).tolist())
    
    # queue_agents_df['agent_movement_order'] = res

    queue_agents_df['agent_movement_order'] = list(map(movement_order.get, queue_agents_df[['current_link', 'next_link']].to_records(index=False).tolist()))
    queue_agents_df['agent_movement_order'] = queue_agents_df['agent_movement_order'].fillna(4) ### if not calculated, set the order to the lowest

    edges_to_use_df = edges_df.loc[edges_df['eid'].isin(queue_agents_df['current_link']) | edges_df['eid'].isin(queue_agents_df['next_link'])]
    edges_storage_remain_dict = dict(zip(edges_to_use_df['eid'], edges_to_use_df['storage_remain']))
    edges_storage_remain_dict[-1] = 1e5
    edges_lanes_dict = dict(zip(edges_to_use_df['eid'], edges_to_use_df['lanes']))
    edges_lanes_dict[-1] = 1e5
    edges_capacity_dict = dict(zip(edges_to_use_df['eid'], edges_to_use_df['in_capacity']))
    edges_capacity_dict[-1] = 1e5
    queue_agents_df['nl_storage_remain'] = queue_agents_df['next_link'].map(edges_storage_remain_dict)
    queue_agents_df['cl_lanes'] = queue_agents_df['current_link'].map(edges_lanes_dict)
    queue_agents_df['nl_lanes'] = queue_agents_df['next_link'].map(edges_lanes_dict)
    queue_agents_df['cl_out_capacity'] = queue_agents_df['current_link'].map(edges_capacity_dict)
    queue_agents_df['nl_in_capacity'] = queue_agents_df['next_link'].map(edges_capacity_dict)

    queue_agents_df['cl_out_capacity_floor'] = np.floor(queue_agents_df['cl_out_capacity'])
    queue_agents_df['cl_out_capacity_ceil'] = np.ceil(queue_agents_df['cl_out_capacity'])
    queue_agents_df['cl_out_capacity_int'] = np.where(np.random.rand(queue_agents_df.shape[0])<(queue_agents_df['cl_out_capacity_ceil'] - queue_agents_df['cl_out_capacity']), queue_agents_df['cl_out_capacity_floor'], queue_agents_df['cl_out_capacity_ceil']).astype(int)

    queue_agents_df['nl_in_capacity_floor'] = np.floor(queue_agents_df['nl_in_capacity'])
    queue_agents_df['nl_in_capacity_ceil'] = np.ceil(queue_agents_df['nl_in_capacity'])
    queue_agents_df['nl_in_capacity_int'] = np.where(np.random.rand(queue_agents_df.shape[0])<(queue_agents_df['nl_in_capacity_ceil']-queue_agents_df['nl_in_capacity']), queue_agents_df['nl_in_capacity_floor'], queue_agents_df['nl_in_capacity_ceil']).astype(int)
    # print(queue_agents_df.groupby('next_link').agg({'nl_in_capacity_int': np.mean}))
    
    ### filter for front agents by output capacity of the current link 
    queue_agents_df['cl_position'] = queue_agents_df.sort_values(by='current_link_enter_time', ascending=True).groupby('current_link').cumcount()
    # queue_agents_df['cl_position'] = queue_agents_df.loc[queue_agents_df['current_link_enter_time'].values.argsort()].groupby('current_link').cumcount()
    # front_agents_df = queue_agents_df.loc[(queue_agents_df['cl_position']<queue_agents_df[['cl_lanes', 'cl_out_capacity_int']].min(axis=1))].copy()
    front_agents_df = queue_agents_df.loc[(queue_agents_df['cl_position']<queue_agents_df['cl_lanes']) & (queue_agents_df['cl_position']<queue_agents_df['cl_out_capacity_int'])].copy()
    # print(queue_agents_df.iloc[0])
    # print(t, queue_agents_df.shape, front_agents_df.shape)

    ### second filter: within next link inflow capacity; within next link storage capacity
    front_agents_df['nl_order'] = front_agents_df.sort_values(by=['agent_movement_order'], ascending=True).groupby('next_link').cumcount()
    front_agents_df = front_agents_df.loc[(front_agents_df['nl_order']<front_agents_df['nl_in_capacity_int']) & (front_agents_df['nl_order']<front_agents_df['nl_storage_remain'])]
    if front_agents_df.shape[0] == 0:
        return agents_df

    ### go vehcles: primary vehicles plus non conflict
    go_agents_df = front_agents_df.copy()
    go_agents_df = go_agents_df.sort_values(by='current_link_enter_time', ascending=True)
    go_agents_df['cn_idx'] = go_agents_df.groupby('cl_enid').cumcount()
    go_agents_df['primary_cl'] = go_agents_df.groupby('cl_enid')['current_link'].transform('first')
    go_agents_df['primary_nl'] = go_agents_df.groupby('cl_enid')['next_link'].transform('first')
    go_agents_df['no_conflict'] = list(map(no_conflict_movement.get, go_agents_df[['primary_cl', 'primary_nl', 'current_link', 'next_link']].to_records(index=False).tolist()))
    ### no conflict if has the same movement as the primary movement
    go_agents_df['no_conflict'] = np.where((go_agents_df['primary_cl']==go_agents_df['current_link']) & (go_agents_df['primary_nl']==go_agents_df['next_link']), 1, go_agents_df['no_conflict'])
    go_agents_df['no_conflict'] = np.where(go_agents_df['next_link']==-1, 1, go_agents_df['no_conflict'])
    go_agents_df = go_agents_df.loc[go_agents_df['no_conflict']==1]
    go_ids = go_agents_df['agent_id']#.values.tolist()
    if go_agents_df.shape[0] == 0:
        return agents_df

    agent_clnl_update = {getattr(agent, 'agent_id'): find_next_link(agent, agent_routes, edge_nid_dict, edge_fft_dict) for agent in go_agents_df.itertuples()}
    agents_df.loc[agents_df['agent_id'].isin(go_ids), ['current_link', 'cl_snid', 'cl_enid', 'cl_fft', 'next_link', 'nl_enid']] = pd.DataFrame(agents_df.loc[agents_df['agent_id'].isin(go_ids), 'agent_id'].map(agent_clnl_update).to_list()).values
    agents_df.loc[agents_df['agent_id'].isin(go_ids), 'onlink_status'] = 1
    agents_df.loc[agents_df['agent_id'].isin(go_ids), 'current_link_enter_time'] = t
    agents_df['agent_status'] = np.where(agents_df['current_link']==-1, -1, agents_df['agent_status'])

    return agents_df

def find_next_link(agent, agent_routes, edge_nid_dict, edge_fft_dict):
    agent_id = getattr(agent, 'agent_id')
    agent_cl_enid = getattr(agent, 'cl_enid')
    ### current link end is destination
    if agent_cl_enid == getattr(agent, 'destin_nid'):
        return [-1, agent_cl_enid, 'sink', 1e5, -2, 'sink2']
    agent_nl_enid = agent_routes[agent_id][agent_cl_enid]
    agent_nl = edge_nid_dict[(agent_cl_enid, agent_nl_enid)]
    agent_nl_fft = edge_fft_dict[agent_nl]
    ### next link end is destination, next next link does not exist
    if agent_nl_enid == getattr(agent, 'destin_nid'):
        return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, -1, 'sink']
    agent_nnl_enid = agent_routes[agent_id][agent_nl_enid]
    agent_nnl = edge_nid_dict[(agent_nl_enid, agent_nnl_enid)]
    return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, agent_nnl, agent_nnl_enid]

def get_agent_routes(origin, destin, g, t):

    sp = g.dijkstra(origin, destin)
    sp_dist = sp.distance(destin)
    if (t==0) and (sp_dist > 1e8):
        sp.clear()
        # self.route = {self.current_link_start_nid: self.current_link_end_nid}
        route = {}
        # self.furthest_nid = self.current_link_end_nid
        status = 'shelter_p'
        # print(self.agent_id, self.current_link_start_nid, self.current_link_end_nid)
    elif (t>0) and (sp_dist>1e8):
        sp.clear()
        pass
    else:
        route_distance = sp_dist
        sp_route = sp.route(destin)
        ### create a path. Order only preserved in Python 3.7+. Do not rely on.
        # self.route = {self.current_link_start_nid: self.current_link_end_nid}
        route = {}
        for (start_nid, end_nid) in sp_route:
            route[start_nid] = end_nid
        sp.clear()
    
    return route

