#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import time 
import random
import logging 
import warnings
import numpy as np
import pandas as pd 
from ctypes import c_double
import scipy.io as sio
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
import scipy.sparse as ssparse
import multiprocessing
from multiprocessing import Pool
from scipy.stats import truncnorm
### dir
home_dir = '.' # os.environ['HOME']+'/spatial_queue'
work_dir = '.' # os.environ['WORK']+'/spatial_queue'
scratch_dir = 'projects/butte_osmnx/simulation_outputs' # os.environ['OUTPUT_FOLDER']
### user
sys.path.insert(0, home_dir+'/..')
from sp import interface
import util.haversine as haversine
from queue_class import Node, Link, Agent

random.seed(0)
np.random.seed(0)

warnings.filterwarnings("error")

### distance to fire starting point
def cnt_veh_in_fire(t_now, veh_loc=None, fire_df=None):
    [veh_lon, veh_lat] = zip(*veh_loc)
    in_fire_dict = dict()
    spread_fire_df = fire_df[fire_df['speed']>0].copy().reset_index(drop=True)
    spread_fire_df['fire_id'] = np.arange(spread_fire_df.shape[0])
    for fire in spread_fire_df.itertuples():
        fire_start_lon, fire_start_lat = getattr(fire, 'lon'), getattr(fire, 'lat')
        fire_speed = getattr(fire, 'speed')
        fire_initial_dist = getattr(fire, 'initial_dist')
        fire_start_time, fire_end_time = getattr(fire, 'start_time'), getattr(fire, 'end_time')
        fire_start_dist = haversine.haversine(np.array(veh_lat), np.array(veh_lon), fire_start_lat, fire_start_lon)
        fire_dist = fire_start_dist - fire_initial_dist - fire_speed*np.min([np.max([t_now, fire_start_time]), fire_end_time]) + fire_speed * fire_start_time
        in_fire_dict[getattr(fire, 'fire_id')] = fire_dist < 0
    in_fire_df = pd.DataFrame(in_fire_dict)
    in_fire_cnt = np.sum(in_fire_df.max(axis=1))
    return in_fire_cnt

def road_closure_time(link_id_dict, fire_df):
    link_geom_df = pd.DataFrame([[link_id, link.geometry] for link_id, link in link_id_dict.items()], columns=['link_id', 'geometry'])
    link_geom_gdf = gpd.GeoDataFrame(link_geom_df, crs='epsg:4326', geometry=link_geom_df['geometry']).to_crs('epsg:26943')
    fire_gdf = gpd.GeoDataFrame(fire_df, crs='epsg:4326', geometry=[Point(xy) for xy in zip(fire_df.lon, fire_df.lat)]).to_crs('epsg:26943')
    link_fire_time_dict = dict()
    for link in link_geom_gdf.itertuples():
        if  (getattr(link, 'link_id') not in [24909, 29119]):
            pass
        link_fire_time = [] ### (distance, fire_speed)
        for fire in fire_gdf.itertuples():
            if  (getattr(link, 'link_id') not in [24909, 29119]) | (getattr(fire, 'type') not in ['pentz', 'initial']):
                pass
            start_lon_proj, start_lat_proj, end_lon_proj,  end_lat_proj  = getattr(link, 'geometry').coords[0][0], getattr(link, 'geometry').coords[0][1], getattr(link, 'geometry').coords[-1][0], getattr(link, 'geometry').coords[-1][1]
            l13 = np.vstack((getattr(fire, 'geometry').coords[0][0] - start_lon_proj, getattr(fire, 'geometry').coords[0][1] - start_lat_proj)).T
            l23 = np.vstack((getattr(fire, 'geometry').coords[0][0] - end_lon_proj, getattr(fire, 'geometry').coords[0][1] - end_lat_proj)).T
            l12 = (end_lon_proj - start_lon_proj, end_lat_proj - start_lat_proj)
            l21 = (start_lon_proj - end_lon_proj, start_lat_proj - end_lat_proj)
            ### line distance
            if l12 == (0, 0):
                ### a loop
                point_line_dist = np.linalg.norm(l13, axis=1)[0]
                # print('loop, fire distance ', point_line_dist)
            else:
                line_dist = np.abs(np.matmul(l13, (l12[-1], -l12[0]))) / np.linalg.norm(l12)
                ### start_dist_array
                start_node_distance = np.linalg.norm(l13, axis=1)
                start_node_angle = np.matmul(l13, l12)
                ### end_dist_array
                end_node_distance = np.linalg.norm(l23, axis=1)
                end_node_angle = np.matmul(l23, l21)
                point_line_dist = np.where(start_node_angle<0, start_node_distance,
                                            np.where(end_node_angle<0, end_node_distance, line_dist))[0]
            
            ### calculate the time that the fire will reach the road
            fire_speed = getattr(fire, 'speed')
            fire_type = getattr(fire, 'type')
            if fire_speed == 0:
                if point_line_dist < getattr(fire, 'initial_dist'):
                    fire_arrival_time = getattr(fire, 'start_time')
                else:
                    fire_arrival_time = 10e7
            else:
                fire_arrival_time = np.max([0, (point_line_dist-getattr(fire, 'initial_dist'))/fire_speed + getattr(fire, 'start_time')])
                if fire_arrival_time > getattr(fire, 'end_time'):
                    fire_arrival_time = 10e7
            if (getattr(link, 'link_id') in [24909, 29119]) & (getattr(fire, 'type') in ['pentz', 'initial']):
                print(getattr(link, 'link_id'))
                print(getattr(fire, 'type'))
                print(point_line_dist)
                print(fire_speed, fire_arrival_time)
                print(start_lon_proj, start_lat_proj, end_lon_proj,  end_lat_proj)
                print('\n')

            link_fire_time.append([point_line_dist, fire_arrival_time, fire_type])
        link_fire_time_dict[getattr(link, 'link_id')] = min(link_fire_time, key=lambda t: t[1]) ### {link_id: (distance, fire_speed)}
    return link_fire_time_dict

# ### distance to fire frontier
# def fire_frontier_distance(fire_frontier, veh_loc, t):
#     [veh_lon, veh_lat] = zip(*veh_loc)
#     if t>=np.max(fire_frontier['t']):
#         fire_frontier_now = fire_frontier.loc[fire_frontier['t'].idxmax(), 'geometry']
#         veh_fire_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_now)
#     else:
#         t_before = np.max(fire_frontier.loc[fire_frontier['t']<=t, 't'])
#         t_after = np.min(fire_frontier.loc[fire_frontier['t']>t, 't'])
#         fire_frontier_before = fire_frontier.loc[fire_frontier['t']==t_before, 'geometry'].values[0]
#         fire_frontier_after = fire_frontier.loc[fire_frontier['t']==t_after, 'geometry'].values[0]
#         veh_fire_dist_before = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_before)
#         veh_fire_dist_after = haversine.point_to_vertex_dist(veh_lon, veh_lat, fire_frontier_after)
#         veh_fire_dist = veh_fire_dist_before * (t_after-t)/(t_after-t_before) + veh_fire_dist_after * (t-t_before)/(t_after-t_before)
#     return np.mean(veh_fire_dist), np.sum(veh_fire_dist<0)
### numbers of vehicles that have left the evacuation zone / buffer distance
def outside_polygon(evacuation_zone, evacuation_buffer, veh_loc):
    [veh_lon, veh_lat] = zip(*veh_loc)
    evacuation_zone_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, evacuation_zone)
    evacuation_buffer_dist = haversine.point_to_vertex_dist(veh_lon, veh_lat, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)

def network(network_file_edges=None, network_file_nodes=None, simulation_outputs=None, cf_files=[], scen_nm=''):
    # logger = logging.getLogger("butte_evac")

    links_df0 = pd.read_csv(work_dir + network_file_edges)
    ### contraflow
    if len(cf_files)>0:
        ### read contraflow links
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
        links_df0['maxmph'] = np.where(links_df0['edge_id_igraph'].isin(opcf_links_id), 0.000001, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 1900*links_df0['lanes']
    links_df0 = links_df0[['eid', 'nid_s', 'nid_e', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]
    links_df0.to_csv(scratch_dir + simulation_outputs + '/modified_network_edges.csv', index=False)
    # sys.exit(0)

    nodes_df0 = pd.read_csv(work_dir + network_file_nodes)

    ### Convert to mtx
    # wgh = links_df0['fft']
    # row = links_df0['nid_s']
    # col = links_df0['nid_e']
    # assert max(np.max(row)+1, np.max(col)+1) == nodes_df0.shape[0], 'nodes and links dimension do not match, row {}, col {}, nodes {}'.format(np.max(row), np.max(col), nodes_df0.shape[0])
    # g_coo = ssparse.coo_matrix((wgh, (row, col)), shape=(nodes_df0.shape[0], nodes_df0.shape[0]))
    # logging.info("({}, {}), {}".format(g_coo.shape[0], g_coo.shape[1], len(g_coo.data)))
    # sio.mmwrite(scratch_dir + simulation_outputs + '/network_sparse_{}.mtx'.format(scen_nm), g_coo)
    # g_coo = sio.mmread(absolute_path+'/outputs/network_sparse.mtx'.format(folder))
    # g = interface.readgraph(bytes(scratch_dir + simulation_outputs + '/network_sparse_{}.mtx'.format('tabu'), encoding='utf-8'))
    g = interface.from_dataframe(links_df0, 'nid_s', 'nid_e', 'fft')

    ### Create link and node objects
    nodes = []
    links = []
    for row in nodes_df0.itertuples():
        real_node = Node(getattr(row, 'nid'), getattr(row, 'lon'), getattr(row, 'lat'), 'real', getattr(row, 'osmid'))
        virtual_node = real_node.create_virtual_node()
        virtual_link = real_node.create_virtual_link()
        nodes.append(real_node)
        nodes.append(virtual_node)
        links.append(virtual_link)
    for row in links_df0.itertuples():
        real_link = Link(getattr(row, 'eid'), getattr(row, 'lanes'), getattr(row, 'length'), getattr(row, 'fft'), getattr(row, 'capacity'), 'real', getattr(row, 'nid_s'), getattr(row, 'nid_e'), getattr(row, 'geometry'))
        links.append(real_link)

    return g, nodes, links


def demand(nodes_osmid_dict, dept_time_df=None, dept_time_col=None, demand_files=None, tow_pct=0, phase_tdiff=None, reroute_pct=0):
    # logger = logging.getLogger("butte_evac")

    all_od_list = []
    for demand_file in demand_files:
        od = pd.read_csv(work_dir + demand_file)
        ### transform OSM based id to graph node id
        od['origin_nid'] = od['origin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        od['destin_nid'] = od['destin_osmid'].apply(lambda x: nodes_osmid_dict[x])
        ### assign agent id
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])
        ### assign departure time. dept_time_std == 0 --> everyone leaves at the same time
        # od = od.merge(dept_time_df, how='left', on='evac_zone')
        od['dept_time'] = od[dept_time_col]
        ### assign vehicle length
        od['veh_len'] = np.random.choice([8, 18], size=od.shape[0], p=[1-tow_pct, tow_pct])
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


def main(random_seed=None, fire_speed=None, dept_time_id=None, dept_time_df=None, dept_time_col=None, tow_pct=None, hh_veh=None, reroute_pct=0, phase_tdiff=None, counterflow=None, fs=None, fire_df=None):
    ### logging and global variables

    global g, agent_id_dict, node_id_dict, link_id_dict, node2link_dict
    fitness = 0
    
    multiprocessing_flag = False
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = '/projects/butte_osmnx/network_inputs/butte_edges_sim.csv'
    network_file_nodes = '/projects/butte_osmnx/network_inputs/butte_nodes_sim.csv'
    demand_files = ["/projects/butte_osmnx/demand_inputs/od.csv"]
    simulation_outputs = '' ### scratch_folder
    if counterflow:
        cf_files = []
    else:
        cf_files = []

    scen_nm = "real_full_c1"
    # logger = logging.getLogger("butte_evac")
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
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
        node.calculate_straight_ahead_links(node_id_dict=node_id_dict, link_id_dict=link_id_dict)
    
    ### demand
    evacuation_zone_gdf = gpd.read_file(work_dir+'/projects/butte_osmnx/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp')
    evacuation_zone_gdf = evacuation_zone_gdf.loc[evacuation_zone_gdf['id']<=14].copy()
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).to_crs('epsg:4326').unary_union
    logging.info('Evacuation zone is {} km2, considering 1mile buffer it is {} km2'.format(evacuation_zone_gdf.to_crs('epsg:3857')['geometry'].unary_union.area/1e6, evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).unary_union.area/1e6))

    agents = demand(nodes_osmid_dict, dept_time_col=dept_time_col, demand_files = demand_files, phase_tdiff=phase_tdiff, reroute_pct=reroute_pct)
    agent_id_dict = {agent.id: agent for agent in agents}

    ### fire arrival time
    link_fire_time_dict = road_closure_time(link_id_dict, fire_df)
    pd.DataFrame([[link_id, link_fire_dist[0], link_fire_dist[1], link_fire_dist[2], link_id_dict[link_id].geometry] for link_id, link_fire_dist in link_fire_time_dict.items()], columns=['link_id', 'link_fire_dist', 'fire_arrival_time', 'fire_type', 'geometry']).to_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm), index=False)
    fire_df.to_csv(scratch_dir + simulation_outputs + '/fire/fire_init_{}.csv'.format(scen_nm), index=False)
    sys.exit(0)
    # link_fire_time_df = pd.read_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm))
    # link_fire_time_dict = {}
    # for row in link_fire_time_df.itertuples():
    #     link_id = getattr(row, 'link_id')
    #     if link_id[0] != 'n':
    #         link_id = int(link_id)
    #     link_fire_time_dict[link_id] = [getattr(row, 'link_fire_dist'), getattr(row, 'fire_arrival_time')]
    
    t_s, t_e = 0, 14401
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'shelter', 'move', 'avg_fdist', 'neg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")
    # with open(scratch_dir + simulation_outputs + '/t_stats/t_shelter_{}.csv'.format(scen_nm), 'w') as t_shelter_outfile:
    #     t_shelter_outfile.write(",".join(['t', 'agent_id', 'agent_origin', 'agent_cls', 'agent_dept_time', 'agent_cl_enter_time'])+"\n")

    closed_links, burnt_links = [], []
    shelter_cnts = 0
    for t in range(t_s, t_e):
        move = 0

        ### agent model
        t_agent_0 = time.time()
        # shelter_list = []
        cl_agents_dict = {cl: [] for cl in closed_links}
        # a1_cnt, a2_cnt, a3_cnt = 0, 0, 0
        for agent_id, agent in agent_id_dict.items():
            if agent.status != 'shelter':
                # a1_cnt += 1
                agent.find_next_link(node2link_dict=node2link_dict)
                try:
                    cl_agents_dict[agent.ol].append(agent_id)
                except KeyError:
                    pass
                ### initial and reroute
                # if (t == 0) | (agent.ol in closed_link):
                if t==0:
                    # a2_cnt += 1
                    routing_status = agent.get_path(g=g)
                    # if routing_status == 'no_route':
                    #     agent.force_stop()
                agent.load_trips(t, node2link_dict=node2link_dict, link_id_dict=link_id_dict)
        for cl_agents in cl_agents_dict.values():
            for agent_id in cl_agents:
                # a3_cnt += 1
                agent = agent_id_dict[agent_id]
                routing_status = agent.get_path(g=g)
                if routing_status == 'no_route':
                    agent.force_stop()
                    shelter_cnts += 1
        t_agent_1 = time.time()
        # print(t, len(closed_links), len(shelter_list), a1_cnt, a2_cnt, a3_cnt, t_agent_1 - t_agent_0)
        # gc.collect()
        # if t==1200: sys.exit(0)

        ### link model
        ### Each iteration in the link model is not time-consuming. So just keep using one process.
        t_link_0 = time.time()
        for link_id, link in link_id_dict.items(): 
            link.run_link_model(t, agent_id_dict=agent_id_dict)
            link.travel_time_list = []
            if (link.type!='v') & (link_id not in burnt_links) & (t > link_fire_time_dict[link_id][1]): ### t larger than the arrival time of the first fire
                burnt_links.append(link_id)
                if (link_fire_time_dict[link_id][2]=='ember') & (random.uniform(0, 1) < 0.1):
                    link.close_link_to_newcomers(g=g)
                    closed_links.append(link_id)
        t_link_1 = time.time()
        
        ### node model
        t_node_0 = time.time()
        node_ids_to_run = set([link.end_nid for link in link_id_dict.values() if len(link.queue_veh)>0])
        for node_id in node_ids_to_run:
            node = node_id_dict[node_id] 
            n_t_move, t_traffic_counter, agent_update_dict, link_update_dict = node.run_node_model(t, node_id_dict=node_id_dict, link_id_dict=link_id_dict, agent_id_dict=agent_id_dict, node2link_dict=node2link_dict)
            move += n_t_move
            for agent_id, agent_new_info in agent_update_dict.items():
                [agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle, agent_id_dict[agent_id].arr_status, agent_id_dict[agent_id].cl_enter_time] = agent_new_info
            for link_id, link_new_info in link_update_dict.items():
                if len(link_new_info) == 3:
                    [link_id_dict[link_id].queue_veh, link_id_dict[link_id].ou_c, link_id_dict[link_id].travel_time_list] = link_new_info
                elif len(link_new_info) == 2:
                    [link_id_dict[link_id].run_veh, link_id_dict[link_id].in_c] = link_new_info
                else:
                    print('invalid link update information')
        t_node_1 = time.time()

        ### metrics
        if t%100 == 0:
            arrival = [agent_id for agent_id, agent in agent_id_dict.items() if agent.arr_status in ['arr']] ### shelter_arr not considered as arrival
            # print(arrival, agent_id_dict[11274].status, agent_id_dict[11274].destin_nid, agent_id_dict[11274].cle)
            arrival_cnts = len(arrival)
            if arrival_cnts + shelter_cnts == len(agent_id_dict):
                logging.info("all agents arrive at destinations")
                break
            veh_loc = [link_id_dict[node2link_dict[(agent.cls, agent.cle)]].midpoint for agent in agent_id_dict.values() if agent.arr_status not in ['arr', 'shelter_arr']]
            # outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
            outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = 0, 0
            avg_fire_dist = 0 # np.mean(fire_point_distance(veh_loc))
            neg_dist = cnt_veh_in_fire(t, veh_loc=veh_loc, fire_df=fire_df)
            with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
                t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, shelter_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")

            ### fitness metric
            fitness += neg_dist
        
        ### stepped outputs
        if t%1200==0:
            link_output = pd.DataFrame([(link.id, len(link.queue_veh), len(link.run_veh), round(link.travel_time, 2)) for link in link_id_dict.values() if link.type=='real'], columns=['link_id', 'q', 'r', 't'])
            link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
            node_agent_cnts = pd.DataFrame([(agent.cle, agent.status, agent.arr_status, 1) for agent in agent_id_dict.values() if (agent.status not in ['flow', 'chance'])], columns=['node_id', 'status', 'arr_status', 'cnt']).groupby(['node_id', 'status', 'arr_status']).agg({'cnt': np.sum}).reset_index()
            node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

        if t%100==0: 
            logging.info(" ".join([str(i) for i in [t, arrival_cnts, shelter_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))

    return fitness


### python3 -c 'import dta_meso_butte; dta_meso_butte.main(random_seed=0, dept_time_id="imm", reroute_pct=0, phase_tdiff=0, counterflow=0)'
if __name__ == "__main__":
    fire_df = pd.read_csv("projects/butte_osmnx/demand_inputs/simulation_fire_locations.csv")
    fire_df['start_time'] = np.where(np.isnan(fire_df['start_time']), np.random.randint(0, 7200, fire_df.shape[0]), fire_df['start_time'])
    fire_df['end_time'] = np.where(np.isnan(fire_df['end_time']), fire_df['start_time'], fire_df['end_time'])
    print(fire_df.head())
    print(fire_df.tail())
    main(random_seed=0, fire_df=fire_df, dept_time_col='dept_time_scen_1')
