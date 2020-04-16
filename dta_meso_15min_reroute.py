#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def network(counterflow=False, closure=False, network_file_edges=None, network_file_nodes=None, simulation_outputs=None, scen_nm=''):

    links_df0 = pd.read_csv(absolute_path+network_file_edges)
    
    links_df0['lanes'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 25, links_df0['maxmph'])

    links_df0['lanes'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 55, links_df0['maxmph'])
    
    if counterflow == True:
#         counterflow_roads = ['euclid_ave', 'spruce_ave', 'grizzly_peak_blvd']
        counterflow_roads = ['marin_ave', 'marin_ave_2', 'laloma_ave', 'college_ave']
        downhill_roads = []
        uphill_roads = []
        for cfr in counterflow_roads:
            cfr_df = pd.read_csv(absolute_path+'/../network/outputs/{}_dir.csv'.format(cfr))
            downhill_roads += cfr_df.loc[cfr_df['downhills']==1, 'edge_id_igraph'].values.tolist()
            uphill_roads = cfr_df.loc[cfr_df['downhills']==0, 'edge_id_igraph'].values.tolist()
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(downhill_roads), 2, links_df0['lanes'])
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(uphill_roads), 0, links_df0['lanes'])
        links_df0['maxmph'] = np.where(links_df0['edge_id_igraph'].isin(uphill_roads), 0.01, links_df0['maxmph'])
    
    if closure == True:
        closure_roads = ['neal_road', 'clark_road', 'pentz_road']
        closure_road_ids = []
        for clr in closure_roads:
            clr_df = pd.read_csv(absolute_path+'/../network/data/butte/osm_edges_{}.csv'.format(clr))
            closure_road_ids += clr_df['edge_id_igraph'].values.tolist()
        links_df0['lanes'] = np.where(links_df0['edge_id_igraph'].isin(closure_road_ids), 0, links_df0['lanes'])
        links_df0['maxmph'] = np.where(links_df0['edge_id_igraph'].isin(closure_road_ids), 0.01, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 2000*links_df0['lanes']
    links_df0['store_cap'] = links_df0['length']*links_df0['lanes']/8 
    links_df0['store_cap'] = np.where(links_df0['store_cap']<1, 1, links_df0['store_cap'])
    links_df0['stype'] = 'real'
    links_df0 = links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'stype', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'store_cap', 'geometry']]
    links_df0.to_csv(absolute_path+simulation_outputs+'/simulation_edges.csv', index=False)

    nodes_df0 = pd.read_csv(absolute_path+network_file_nodes)
    nodes_df0['node_id_sp'] = nodes_df0['node_id_igraph'] + 1

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

    return g, links_df0, nodes_df0


# In[3]:


def demand(nodes_df0, phased_flag = False, demand_files=None):
    
    if demand_files == None:
        o_sp = np.random.randint(low=0, high=np.max(nodes_df0['node_id_igraph']), size = 1000) + 1
        d_sp = np.random.randint(low=0, high=np.max(nodes_df0['node_id_igraph']), size = 1000) + 1
        agent_id = np.arange(1000)
        agent_departure_time = np.random.randint(low=0, high=100, size=1000)
        od = pd.DataFrame({'agent_id': agent_id, 'o_sp': o_sp, 'd_sp': d_sp, 'd_tm': agent_departure_time})
        od['cls'] = od['o_sp'].apply(lambda x: 'vn{}'.format(x-1)) ### id of the start node of the current link
        od['cle'] = od['o_sp'] - 1 ### id of the end node of the current link
#         od = pd.DataFrame([[0,3024,14902,0,'vn3023',3023]], columns=['agent_id', 'o_sp', 'd_sp', 'd_tm', 'cls', 'cle'])
        return od
    
    else:
        all_od_list = []
        for demand_file in demand_files:
            od = pd.read_csv(absolute_path + demand_file)
            
            if 'agent_id' not in od.columns:
                od['agent_id'] = np.arange(od.shape[0])
                
            if phased_flag == False:
                od['d_tm'] = 0
            else:
                od['d_tm'] = np.random.randint(low=0, high=3600*5, size=od.shape[0])
            
            od = pd.merge(od, nodes_df0[['node_id_igraph', 'node_osmid']], how='left', left_on='origin_osmid', right_on='node_osmid')
            od['o_sp'] = od['node_id_igraph'] + 1
            od = pd.merge(od[['agent_id', 'o_sp', 'destin_osmid', 'd_tm']], nodes_df0[['node_id_igraph', 'node_osmid']], how='left', left_on='destin_osmid', right_on='node_osmid')
            od['d_sp'] = od['node_id_igraph'] + 1
            all_od_list.append(od)
        all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
        all_od['cls'] = all_od['o_sp'].apply(lambda x: 'vn{}'.format(x-1)) ### id of the start node of the current link
        all_od['cle'] = all_od['o_sp'] - 1 ### id of the end node of the current link
        all_od = all_od[['agent_id', 'o_sp', 'd_sp', 'd_tm', 'cls', 'cle']]
        all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
        print('total numbers of agents from file ', all_od.shape)
        all_od = all_od.iloc[0:5000].copy()
        
    print('total numbers of agents taken ', all_od.shape)
    print(all_od.head())
        
    return all_od


# In[4]:


def map_sp(agent_id):
    
    ### Find shortest path for each unique origin --> one destination
    origin_ID = agent_info[agent_id]['o_sp']
    destin_ID = agent_info[agent_id]['d_sp']
    depart_time = agent_info[agent_id]['d_tm']
    current_link_start = agent_info[agent_id]['cls']
    current_link_end = agent_info[agent_id]['cle']
    
    sp = g.dijkstra(current_link_end+1, destin_ID)
    sp_dist = sp.distance(destin_ID) ### agent believed travel time with imperfect information
    
    if sp_dist > 10e7:
        sp_edges = []
        results = {'agent_id': agent_id, 'route_igraph': sp_edges}
        sp.clear()
        return results, 'n_a'
    else:
        sp_route = sp.route(destin_ID)
        path = [(current_link_start, current_link_end)] + [(start_sp-1, end_sp-1) for (start_sp, end_sp) in sp_route]
        sp.clear()
#         print('map', agent_info[0])
#         print('map', current_link_end, destin_ID, path)
        results = {'agent_id': agent_id, 'cls': current_link_start, 'cle': current_link_end, 'o_sp': origin_ID, 'd_sp': destin_ID, 'd_tm': depart_time, 'route_igraph': path}
#         print(results, current_link_start, current_link_end)
        ### [(edge[0], edge[1]) for edge in sp_route]: agent's choice of route
        return results, 'a' ### 'a' means arrival
    
def reduce_edge_flow(agent_info_routes):
    ### Reduce (count the total traffic flow per edge) with pandas groupby

    flat_L = [(e[0], e[1]) for r in agent_info_routes for e in r['route_igraph'] if len(r['route_igraph'])>0]
    df_L = pd.DataFrame(flat_L, columns=['start_igraph', 'end_igraph'])
    df_L_flow = df_L.groupby(['start_igraph', 'end_igraph']).size().reset_index().rename(columns={0: 'vol'})
    
    return df_L_flow

def route(links_df0, counterflow=False, scen_nm='', simulation_outputs=None):

    if len(agent_info) == 0:
        return {}
    
    ### Build a pool
    process_count = 10
    pool = Pool(processes=process_count)

    ### Find shortest pathes
    t_odsp_0 = time.time()
    res = pool.imap_unordered(map_sp, agent_info.keys())

    ### Close the pool
    pool.close()
    pool.join()
    t_odsp_1 = time.time()

    agent_info_routes, destination_counts = zip(*res)
    edge_volume = reduce_edge_flow(agent_info_routes)
    # print(edge_volume.describe())
    edge_volume = pd.merge(links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'capacity', 'geometry']], edge_volume, how='left', on=['start_igraph', 'end_igraph'])
    edge_volume = edge_volume.fillna(value={'vol': 0})
    ### voc
    edge_volume['voc'] = edge_volume['vol']/edge_volume['capacity']
    edge_volume = edge_volume.sort_values(by='voc', ascending=False)
#     edge_volume[['edge_id_igraph', 'start_igraph', 'end_igraph', 'geometry', 'vol', 'voc']].to_csv(absolute_path+simulation_outputs+'/initial_route_volume_a{}_{}.csv'.format(len(agent_info_routes), scen_nm), index=False)

    cannot_arrive = np.sum([1 for i in destination_counts if i=='n_a'])
    # print('{} out of {} cannot arrive.'.format(cannot_arrive, len(agent_info)))
#     print('routing takes {} sec'.format(t_odsp_1 - t_odsp_0))

    new_agent_info = {a['agent_id']: {'o_sp': a['o_sp'], 'd_sp': a['d_sp'], 'd_tm': a['d_tm'], 'cls': a['cls'], 'cle': a['cle'], 'route_igraph': a['route_igraph']} for a in agent_info_routes if len(a['route_igraph'])>0}
    
    return new_agent_info


# In[5]:


def update_graph(links_attr_dict=None, links_trav_time_dict=None, link_time_lookback_freq=None):
    ### Update graph

    t_update_0 = time.time()
    
    new_links_trav_time_dict = {}
    avg_links_trav_time = []
    for k, v in links_trav_time_dict.items():
        recent_v = [(t_rec, dur) for (t_rec, dur) in v if (t-t_rec < link_time_lookback_freq)]
        if len(recent_v) == 0:
            pass
        else:
            new_links_trav_time_dict[k] = recent_v
            avg_links_trav_time.append((k, np.avg([dur for (_, dur) in recent_v])))

    if len(avg_links_trav_time) == 0:
        pass
    else:
        for (link_id, avg_trav_time) in avg_links_trav_time:
            g.update_edge(links_attr_dict[link_id]['s_i']+1, links_attr_dict[link_id]['e_i']+1, c_double(avg_trav_time))

    t_update_1 = time.time()
#     print('updating graph takes {} sec'.format(t_update_1 - t_update_0))

    return new_links_trav_time_dict


# In[6]:


def virtual_nodes_links(links_df0, nodes_df0):

    virtual_nodes_df = nodes_df0.copy()
    virtual_nodes_df['node_id_igraph'] = virtual_nodes_df['node_id_igraph'].apply(lambda x: 'vn{}'.format(x))
    virtual_nodes_df['node_id_sp'] = virtual_nodes_df['node_id_sp'].apply(lambda x: 'vn{}_sp'.format(x))
    virtual_nodes_df['lon'] = virtual_nodes_df['lon'] + 0.001
    virtual_nodes_df['lat'] = virtual_nodes_df['lat'] + 0.001
    nodes_df = pd.concat([nodes_df0, virtual_nodes_df], sort=False, ignore_index=True)

    virtual_links_dict = {'edge_id_igraph':[], 'start_igraph':[], 'end_igraph':[], 'stype': [], 'lanes': [], 'capacity':[], 'fft':[], 'length':[], 'store_cap':[], 'geometry':[]}
    for node in nodes_df0.itertuples():
        node_id = getattr(node, 'node_id_igraph')
        node_lon = getattr(node, 'lon')
        node_lat = getattr(node, 'lat')

        virtual_links_dict['edge_id_igraph'].append('n{}_vl'.format(node_id))
        virtual_links_dict['start_igraph'].append('vn{}'.format(node_id))
        virtual_links_dict['end_igraph'].append(node_id)
        virtual_links_dict['stype'].append('v')
        virtual_links_dict['lanes'].append(100)
        virtual_links_dict['capacity'].append(100000)
        virtual_links_dict['fft'].append(0)
        virtual_links_dict['length'].append(0)
        virtual_links_dict['store_cap'].append(100000)
        virtual_links_dict['geometry'].append('LINESTRING({} {}, {} {})'.format(node_lon+0.001, node_lat+0.001, node_lon, node_lat))
    virtual_links_df = pd.DataFrame(virtual_links_dict)
    links_df = pd.concat([links_df0, virtual_links_df], sort=False, ignore_index=True)

    return links_df, nodes_df


# In[7]:


def sending_receiving(t, t_scale, links_dict=None, links_attr_dict=None):

    t_sending_receiving_0 = time.time()
    new_links_dict = {}
    for l_id, l_traf in links_dict.items():
        l_traf_run_new = []
        l_traf_queue_new = l_traf['queue']
        for [agent, t_enter] in l_traf['run']:
            if t_enter < t*t_scale - links_attr_dict[l_id]['fft']:
                l_traf_queue_new.append([agent, t_enter])
            else:
                l_traf_run_new.append([agent, t_enter])
        l_traf_sending_new = links_attr_dict[l_id]['ou_c']/3600*t_scale
        l_traf_receiving_new = links_attr_dict[l_id]['in_c']/3600*t_scale
        l_traf_store_cap_remain = links_attr_dict[l_id]['st_c'] - len(l_traf_run_new) - len(l_traf_queue_new) ###? storage cap does not change with time slice size, but sending and receiving cap change with time slice size
        new_links_dict[l_id] = {'run': l_traf_run_new, 'queue': l_traf_queue_new, 'send': l_traf_sending_new, 'receive': l_traf_receiving_new, 'st_remain': l_traf_store_cap_remain}
    t_sending_receiving_1 = time.time()
#     print('link model time {} sec'.format(t_sending_receiving_1 - t_sending_receiving_0))
    
    return new_links_dict


# In[8]:


def nodal_transfer(t=0, t_scale=1, nodes_dict=None, links_dict=None, links_attr_dict=None, links_trav_time_dict=0, node2edge=None, reroute_flag=False):

    node_transfer_0 = time.time()
    arrival_list = []
    move = 0

    for n, in_out in nodes_dict.items():

        in_links = in_out['in_links'].keys()
        out_links = in_out['out_links']
        x_mid = in_out['lon']
        y_mid = in_out['lat']

        in_links = [l for l in in_links if len(links_dict[l]['queue'])>0]
        if len(in_links) == 0:
            continue

        go_link = random.choice(in_links)
        x_start = nodes_dict[links_attr_dict[go_link]['s_i']]['lon']
        y_start = nodes_dict[links_attr_dict[go_link]['s_i']]['lat']
        in_vec = (x_mid-x_start, y_mid-y_start)
        go_vehs = []
        left_turn_vehs = False
        incoming_lanes = int(np.floor(links_attr_dict[go_link]['ln']))
        incoming_vehs = len(links_dict[go_link]['queue'])
        for ln in range(min(incoming_lanes, incoming_vehs)):
            [agent_id, link_enter_time] = links_dict[go_link]['queue'][ln]
            try:
                agent_next_node = [end for (start, end) in agent_info[agent_id]['route_igraph'] if start == n][0]
            except IndexError:
                go_vehs.append([agent_id, None, go_link, None, link_enter_time])
                left_turn_vehs = False or left_turn_vehs
                continue

            ol = node2edge[(n, agent_next_node)]
            go_vehs.append([agent_id, agent_next_node, go_link, ol, link_enter_time])
            if links_attr_dict[go_link]['ty']=='v': ### virtual enter
                left_turn_vehs = False or left_turn_vehs
            else:
                x_end = nodes_dict[agent_next_node]['lon']
                y_end = nodes_dict[agent_next_node]['lat']
                out_vec = (x_end-x_mid, y_end-y_mid)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                agent_dir = np.arctan2(det, dot)*180/np.pi 
                if agent_dir < -45:
                    left_turn_vehs = True or left_turn_vehs
        
        op_go_vehs = []
        if (not left_turn_vehs) and (links_attr_dict[go_link]['ty']=='real'):
            op_go_link = nodes_dict[n]['in_links'][go_link]
            if op_go_link == None:
                pass
            else:
                x_start = nodes_dict[links_attr_dict[go_link]['s_i']]['lon']
                y_start = nodes_dict[links_attr_dict[go_link]['s_i']]['lat']
                in_vec = (x_mid-x_start, y_mid-y_start)
                op_incoming_lanes = int(np.floor(links_attr_dict[op_go_link]['ln']))
                op_incoming_vehs = len(links_dict[op_go_link]['queue'])
                for ln in range(min(op_incoming_lanes, op_incoming_vehs)):
                    [agent_id, link_enter_time] = links_dict[op_go_link]['queue'][ln]
                    try:
                        agent_next_node = [end for (start, end) in agent_info[agent_id]['route_igraph'] if start == n][0]
                    except IndexError:
                        op_go_vehs.append([agent_id, None, op_go_link, None, link_enter_time])
                        continue
                    ol = node2edge[(n, agent_next_node)]
                    x_end = nodes_dict[agent_next_node]['lon']
                    y_end = nodes_dict[agent_next_node]['lat']
                    out_vec = (x_end-x_mid, y_end-y_mid)
                    dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                    det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                    agent_dir = np.arctan2(det, dot)*180/np.pi 
                    if agent_dir > 45:
                        op_go_vehs.append([agent_id, agent_next_node, op_go_link, ol, link_enter_time])
                    elif agent_dir > -45:
                        op_go_vehs.append([agent_id, agent_next_node, op_go_link, ol, link_enter_time])
                    else:
                        pass ### no left turn allowed for opposite lane "bonus movement"
                
        for go_vehs_list in [go_vehs, op_go_vehs]:
            for [agent_id, next_node, il, ol, link_enter_time] in go_vehs_list:

                ### Agent reaching destination
                if (next_node is None) and (n == agent_info[agent_id]['d_sp']-1):
                    del agent_info[agent_id]
                    arrival_list.append([agent_id, t])
                    links_dict[go_link]['queue'] = [v for v in links_dict[go_link]['queue'] if v[0]!=agent_id]
                    links_dict[go_link]['send'] = max(0, links_dict[go_link]['send']-1)
                    try:
                        links_trav_time_dict[go_link].append((t, t*t_scale-link_enter_time))
                    except KeyError:
                        pass
                    continue
                
                ### no storage capacity downstream
                if links_dict[ol]['st_remain'] < 1:
                    pass ### no blocking, as # veh = # lanes
                ### inlink-sending, outlink-receiving both permits
                elif (links_dict[il]['send'] >= 1) & (links_dict[ol]['receive'] >= 1):
                    move += 1
                    agent_info[agent_id]['cls'] = n
                    agent_info[agent_id]['cle'] = next_node
                    links_dict[il]['queue'] = [v for v in links_dict[il]['queue'] if v[0]!=agent_id]
                    links_dict[il]['send'] -= 1 ### guaranted larger than 0
                    links_dict[ol]['run'].append((agent_id, t*t_scale))
                    links_dict[ol]['receive'] -= 1 ### guaranted larger than 
                    try:
                        links_trav_time_dict[il].append((t, t*t_scale-link_enter_time))
                    except KeyError:
                        pass
                else: ### either inlink-sending or outlink-receiving or both exhaust
                    control_cap = min(links_dict[il]['send'], links_dict[ol]['receive'])
                    toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                    if toss_coin[0]:
                        move += 1
                        agent_info[agent_id]['cls'] = n
                        agent_info[agent_id]['cle'] = next_node
                        links_dict[il]['queue'] = [v for v in links_dict[il]['queue'] if v[0]!=agent_id]
                        links_dict[il]['send'] = max(0, links_dict[il]['send']-1)
                        links_dict[ol]['run'].append((agent_id, t*t_scale))
                        links_dict[ol]['receive'] = max(0, links_dict[ol]['receive']-1)
                        try:
                            links_trav_time_dict[il].append((t, t*t_scale-link_enter_time))
                        except KeyError:
                            pass
                    else:
                        pass
    
#     print(t, agent_info[0])
    node_transfer_1 = time.time()
#     print('node model time {}'.format(node_transfer_1 - node_transfer_0))
    
    return links_dict, links_trav_time_dict, arrival_list, move, agent_info


# In[9]:


def load_trips(t, t_scale, links_dict, node2edge):

    for a, info in agent_info.items():
        if (info['d_tm'] == t):
            initial_edge = node2edge[info['route_igraph'][0]]
            links_dict[initial_edge]['run'].append([a, t*t_scale])
        
    return links_dict


# In[10]:


def output_interpolated_positions(t=0, links_dict=None, links_attr_dict=None, simulation_outputs=None):
    
    output_interpolated_0 = time.time()
    veh_loc = []
    for l_id, l in links_dict.items():
        if links_attr_dict[l_id]['ty'] == 'v':
            continue
        l_len = links_attr_dict[l_id]['len']
        l_ln = links_attr_dict[l_id]['ln']
        l_fft = links_attr_dict[l_id]['fft']
        l_geom = loads(links_attr_dict[l_id]['geom'])
        q_v_loc = l_len
        q_v_loc_count = 0
        run_veh = sorted(l['run'],key=itemgetter(1))
        queue_veh = sorted(l['queue'],key=itemgetter(1))
        for q_v in queue_veh:
            q_v_loc = max(min(q_v_loc*(1), l_len), 0)
            q_v_coord = l_geom.interpolate(q_v_loc/l_len, normalized=True)
            veh_loc.append([t, l_id, q_v[0], 'q', q_v_coord.x, q_v_coord.y])
            q_v_loc_count += 1
            if q_v_loc_count == l_ln:
                q_v_loc -= 8
                q_v_loc_count = 0
        queue_loc = q_v_loc
        for r_v in run_veh:
            if l_len*(t-r_v[1])/l_fft>queue_loc:
                r_v_loc = queue_loc
                q_v_loc_count += 1
                if q_v_loc_count == l_ln:
                    q_v_loc -= 8
                    q_v_loc_count = 0
            else:
                r_v_loc = l_len*(t-r_v[1])/l_fft
            r_v_loc = max(min(r_v_loc*(1), l_len), 0)
            r_v_coord = l_geom.interpolate(r_v_loc/l_len, normalized=True)
            veh_loc.append([t, l_id, r_v[0], 'r', r_v_coord.x, r_v_coord.y])
    
    pd.DataFrame(veh_loc, columns=['t_sec', 'l_id', 'v_id', 'status', 'lon', 'lat']).to_csv(absolute_path + simulation_outputs + '/veh_loc/veh_loc_{}s.csv'.format(t), index=False)
    output_interpolated_1 = time.time()
#     print('output interpolated {} sec'.format(output_interpolated_1 - output_interpolated_0))


# In[4]:


def main():
    random.seed(0)
    np.random.seed(0)
    global g
    global agent_info
    
    reroute_flag = False
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    counterflow_flag = False
    closure_flag = False
    phased_flag = False
    scen_nm = '3_per_origin_nrr'
    network_file_edges = '/projects/bolinas_stinson_beach/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/bolinas_stinson_beach/network_inputs/osm_nodes.csv'
    demand_files = ['/projects/bolinas_stinson_beach/demand_inputs/bolinas_od_3_per_origin.csv']
    simulation_outputs = '/projects/bolinas_stinson_beach/simulation_outputs'

    t_scale = 1

    g, links_df0, nodes_df0 = network(
        counterflow = counterflow_flag, 
        closure = closure_flag, 
        network_file_edges = network_file_edges,
        network_file_nodes = network_file_nodes,
        simulation_outputs = simulation_outputs,
        scen_nm = scen_nm)
    # return
    od = demand(nodes_df0, 
        phased_flag = phased_flag,
        demand_files = demand_files)
    
    links_df, nodes_df = virtual_nodes_links(links_df0, nodes_df0)
    print(links_df.shape, nodes_df.shape, links_df0.shape, nodes_df0.shape)
    
    node2edge = {(getattr(e, 'start_igraph'), getattr(e, 'end_igraph')): getattr(e, 'edge_id_igraph') for e in links_df.itertuples()}

    links_attr_dict = {getattr(e, 'edge_id_igraph'): {'fft': getattr(e, 'fft'), 'len': getattr(e, 'length'), 'ty': getattr(e, 'stype'), 'ln': getattr(e, 'lanes'), 's_i': getattr(e, 'start_igraph'), 'e_i': getattr(e, 'end_igraph'), 'geom': getattr(e, 'geometry'), 'in_c': getattr(e, 'capacity'), 'ou_c': getattr(e, 'capacity'), 'st_c': getattr(e, 'store_cap')} for e in links_df.itertuples()}

    ### signal at entrance to chico
    # links_attr_dict[21044]['ou_c'] /= 2
    links_dict = {e: {'run': [], 'queue': []} for e in links_df['edge_id_igraph'].values.tolist()}
    links_trav_time_dict = {e: [] for e in links_df.loc[links_df['stype']=='real', 'edge_id_igraph'].values.tolist()}

    nodes_dict = {getattr(n, 'node_id_igraph'): {'in_links': {}, 'out_links': [], 'lon': getattr(n, 'lon'), 'lat': getattr(n, 'lat')} for n in nodes_df.itertuples()}
    for l in links_df.itertuples():
        nodes_dict[getattr(l, 'start_igraph')]['out_links'].append(getattr(l, 'edge_id_igraph'))
        nodes_dict[getattr(l, 'end_igraph')]['in_links'][getattr(l, 'edge_id_igraph')] = None
    for n, in_out in nodes_dict.items():
        x_mid = in_out['lon']
        y_mid = in_out['lat']
        for il in in_out['in_links'].keys():
            x_start = nodes_dict[links_attr_dict[il]['s_i']]['lon']
            y_start = nodes_dict[links_attr_dict[il]['s_i']]['lat']
            in_vec = (x_mid-x_start, y_mid-y_start)
            sa_ol = None
            ol_dir = 180
            for ol in in_out['out_links']:
                x_end = nodes_dict[links_attr_dict[ol]['e_i']]['lon']
                y_end = nodes_dict[links_attr_dict[ol]['e_i']]['lat']
                out_vec = (x_end-x_mid, y_end-y_mid)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                new_ol_dir = np.arctan2(det, dot)*180/np.pi
                if abs(new_ol_dir)<ol_dir:
                    sa_ol = ol
                    ol_dir = new_ol_dir
            if (abs(ol_dir)<=45) and links_attr_dict[il]['ty']=='real':
                nodes_dict[n]['in_links'][il] = sa_ol

    total_arrival_count = 0
    total_arrival_list = []
    agent_info = {}
    track_edges = [295, 543, 596]
    track_edge_info = []
    
    t_s = 0
    t_e = 3600
    for t in range(t_s, t_e):
        ### calculate the paths for agents moving in the next 60 seconds
        if (t==0) or (reroute_flag) and (t%reroute_freq == 0):
            ### add new agents that are scheduled to leave in the next reroute period
            ### to the remaining agents
            new_od = od.loc[(od['d_tm']>=t//reroute_freq*reroute_freq) & (od['d_tm']<(t//reroute_freq+1)*reroute_freq)]
            for row in new_od.itertuples():
                agent_info[getattr(row, 'agent_id')] = {'o_sp': getattr(row, 'o_sp'), 'd_sp': getattr(row, 'd_sp'), 'd_tm': getattr(row, 'd_tm'), 'cls': getattr(row, 'cls'), 'cle': getattr(row, 'cle')}
            # print('{} reroutes at time {}, {} newly added'.format(len(agent_info), t, new_od.shape[0]))
            ### update link travel time
            links_trav_time_dict = update_graph(links_attr_dict = links_attr_dict, links_trav_time_dict=links_trav_time_dict, link_time_lookback_freq = link_time_lookback_freq)
            ### route
            agent_info = route(links_df0, counterflow=counterflow_flag, simulation_outputs = simulation_outputs, scen_nm = scen_nm)

        ### load trips onto the network
        links_dict = load_trips(t, t_scale, links_dict, node2edge)
        ### link model
        links_dict = sending_receiving(t, t_scale, links_dict=links_dict, links_attr_dict=links_attr_dict)
        # output_interpolated_positions(t=t, links_dict=links_dict, links_attr_dict=links_attr_dict, simulation_outputs=simulation_outputs)
        ### node model
        links_dict, links_trav_time_dict, arrival_list, move, agent_info = nodal_transfer(t=t, t_scale=t_scale, nodes_dict=nodes_dict, links_dict=links_dict, links_attr_dict=links_attr_dict, links_trav_time_dict=links_trav_time_dict, node2edge=node2edge, reroute_flag=reroute_flag)
        total_arrival_count += len(arrival_list)
        queue_veh = sum([len(l['queue']) for l in links_dict.values()])
        queue_link = len([l for l in links_dict.values() if len(l['queue'])>0])
        run_link = len([l for l in links_dict.values() if len(l['run'])>0])
        total_arrival_list.append([t, total_arrival_count, move, queue_veh, queue_link, run_link])

        if t%500==0:
            print(total_arrival_list[-1])
            pd.DataFrame([[l, len(v['run']), len(v['queue']), links_attr_dict[l]['geom']] for l, v in links_dict.items()], columns=['edge_id_igraph', 'run', 'queue', 'geom']).to_csv(absolute_path+simulation_outputs+'/run_queue/run_queue_{}_{}s.csv'.format(scen_nm, t), index=False)
        
        ### occupancy of selected edges
        for t_e in track_edges:
            track_edge_info.append([t, t_e, l(links_dict[t_e]['run']), l(links_dict[t_e]['queue'])])


    pd.DataFrame(total_arrival_list, columns=['t_sec', 'tot_arr', 'move', 'q_veh', 'q_l', 'r_l']).to_csv(absolute_path+simulation_outputs+'/arrival_counts/arrival_counts_{}_{}s_{}s.csv'.format(scen_nm, t_s, t_e))


# In[11]:


main()


# In[ ]:


get_ipython().system(u'jupyter nbconvert --to script dta_meso_15min_reroute.ipynb')


# In[ ]:




