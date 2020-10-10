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
from queue_class import Network, Node, Link, Agent

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
    ### https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
    link_geom_df = pd.DataFrame([[link_id, link.geometry] for link_id, link in link_id_dict.items() if link.link_type != 'v'], columns=['link_id', 'geometry'])
    link_geom_gdf = gpd.GeoDataFrame(link_geom_df, crs='epsg:4326', geometry=link_geom_df['geometry']).to_crs('epsg:3857')
    fire_gdf = gpd.GeoDataFrame(fire_df, crs='epsg:4326', geometry=[Point(xy) for xy in zip(fire_df.lon, fire_df.lat)]).to_crs('epsg:3857')

    offset_1, offset_2 = 100, 10000
    fire_gdf['offset'] = np.where(fire_gdf['speed']==0, offset_1, offset_2)
    bbox = fire_gdf.bounds
    bbox['minx'] -= fire_gdf['offset']
    bbox['miny'] -= fire_gdf['offset']
    bbox['maxx'] += fire_gdf['offset']
    bbox['maxy'] += fire_gdf['offset']

    hits = bbox.apply(lambda row: list(link_geom_gdf.sindex.intersection(row)), axis=1)

    tmp = pd.DataFrame({
        # index of points table
        "pt_idx": np.repeat(hits.index, hits.apply(len)),    # ordinal position of line - access via iloc later
        "line_i": np.concatenate(hits.values)
    })
    tmp = tmp.join(link_geom_gdf.reset_index(drop=True), on="line_i")
    tmp = tmp.join(fire_gdf.reset_index(drop=True), on="pt_idx", lsuffix='_l', rsuffix='_pt')
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry_l", crs=fire_gdf.crs)
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.geometry_pt))
    tmp = tmp.loc[(tmp.snap_dist <= tmp.initial_dist) | (tmp.initial_dist==0)]
    tmp["fire_arr"] = np.where(tmp['speed']==0, tmp['start_time'], tmp['snap_dist']/tmp['speed'])
    # print(tmp[tmp['link_id']==9420])
    tmp["fire_arr"] = np.minimum(tmp['end_time'], np.maximum(tmp['fire_arr'], tmp['start_time']))
    # print(tmp[tmp['link_id']==9420])

    tmp = tmp.sort_values(by='fire_arr', ascending=True).groupby('link_id').first().reset_index()
    link_fire_time_dict = {getattr(row, 'link_id'): [getattr(row, 'snap_dist'), getattr(row, 'fire_arr'), getattr(row, 'type')] for row in tmp.itertuples()}

    # print(link_fire_time_dict[9420])

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

def link_model(t, network, link_fire_time_dict, burnt_links, closed_links):
    for link_id, link in network.links.items(): 
        link.run_link_model(t, agent_id_dict=network.agents)
        if link.link_type == 'v': ### do not track the link time of virtual links
            link.travel_time_list = []
        else:
            link.travel_time_list = []
        # link.update_travel_time(self, t_now, link_time_lookback_freq=None, g=None, update_graph=False)
        if (link_id in link_fire_time_dict.keys() ) and (link_id not in burnt_links) and (t > link_fire_time_dict[link_id][1]): ### t larger than the arrival time of the first fire
            burnt_links.append(link_id)
            if (link_fire_time_dict[link_id][2] in ['ember', 'initial']) and (random.uniform(0, 1) < 0.1):
                link.close_link_to_newcomers(g=network.g)
                closed_links.append(link_id)
            elif (link_fire_time_dict[link_id][2] in ['pentz', 'neal', 'clark']):
                link.close_link_to_newcomers(g=network.g)
                closed_links.append(link_id)
            else:
                pass
    return network, burnt_links, closed_links

def node_model(t, network, move, check_traffic_flow_links_dict):
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles)>0])
    for node_id in node_ids_to_run:
        node = network.nodes[node_id] 
        n_t_move, transfer_links, agent_update_dict, link_update_dict = node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links, agent_id_dict=network.agents, node2link_dict=network.node2link_dict)
        move += n_t_move
        ### how many people moves across a specified link pair in this step
        for transfer_link in transfer_links:
            if transfer_link in check_traffic_flow_links_dict.keys():
                check_traffic_flow_links_dict[transfer_link] += 1
        for agent_id, agent_new_info in agent_update_dict.items():
            ### move stopped vehicles to a separate dictionary
            if agent_new_info[0] in ['shelter_arrive', 'arrive']:
                network.agents_stopped[agent_id] = agent_new_info
                del network.agents[agent_id]
            ### update vehicles still not stopped
            else:
                agent = network.agents[agent_id]
                [agent.status, agent.current_link_start_nid, agent.current_link_end_nid,  agent.current_link_enter_time] = agent_new_info
                agent.find_next_link(t, node2link_dict = network.node2link_dict)

        for link_id, link_new_info in link_update_dict.items():
            if link_new_info[0] == 'inflow':
                [_, network.links[link_id].queue_vehicles, 
                network.links[link_id].remaining_outflow_capacity, 
                network.links[link_id].travel_time_list] = link_new_info
            elif link_new_info[0] == 'outflow':
                [_, network.links[link_id].run_vehicles, 
                network.links[link_id].remaining_inflow_capacity,
                network.links[link_id].remaining_storage_capacity] = link_new_info
            else:
                print('invalid link update information')
        
    return network, move, check_traffic_flow_links_dict

def main(random_seed=None, fire_speed=None, dept_time_id=None, dept_time_df=None, dept_time_col=None, tow_pct=None, hh_veh=None, reroute_pct=0, phase_tdiff=None, counterflow=None, fs=None, fire_df=None):
    ### logging and global variables

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

    scen_nm = "full_dict_c0.1_d1"
    # logger = logging.getLogger("butte_evac")
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)

    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = [(29,33)]
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'w') as transfer_stats_outfile:
        transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    ### network
    network = Network()
    network.dataframe_to_network(network_file_edges = network_file_edges, network_file_nodes = network_file_nodes)
    network.add_connectivity()

    ### demand
    network.add_demand(dept_time_col=dept_time_col, demand_files = demand_files, phase_tdiff=phase_tdiff, reroute_pct=reroute_pct)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    ### evacuation zone
    evacuation_zone_gdf = gpd.read_file(work_dir+'/projects/butte_osmnx/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp')
    evacuation_zone_gdf = evacuation_zone_gdf.loc[evacuation_zone_gdf['id']<=14].copy()
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).to_crs('epsg:4326').unary_union
    logging.info('Evacuation zone is {} km2, considering 1mile buffer it is {} km2'.format(evacuation_zone_gdf.to_crs('epsg:3857')['geometry'].unary_union.area/1e6, evacuation_zone_gdf.to_crs('epsg:3857').buffer(1609).unary_union.area/1e6))

    ### fire arrival time
    link_fire_time_dict = road_closure_time(network.links, fire_df)
    pd.DataFrame([[link_id, link_fire_dist[0], link_fire_dist[1], link_fire_dist[2], network.links[link_id].geometry] for link_id, link_fire_dist in link_fire_time_dict.items()], columns=['link_id', 'link_fire_dist', 'fire_arrival_time', 'fire_type', 'geometry']).to_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm), index=False)
    fire_df.to_csv(scratch_dir + simulation_outputs + '/fire/fire_init_{}.csv'.format(scen_nm), index=False)
    # sys.exit(0)
    link_fire_time_df = pd.read_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm))
    link_fire_time_dict = {}
    for row in link_fire_time_df.itertuples():
        link_id = getattr(row, 'link_id')
        link_fire_time_dict[link_id] = [getattr(row, 'link_fire_dist'), getattr(row, 'fire_arrival_time'), getattr(row, 'fire_type')]
    print(link_fire_time_dict[link_id])
    
    t_s, t_e = 0, 18001
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'shelter', 'move', 'avg_fdist', 'neg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")

    closed_links, burnt_links = [], []
    for t in range(t_s, t_e):
        
        move = 0
        ### agent model
        t_agent_0 = time.time()
        for agent_id, agent in network.agents.items():
            # initial route 
            if t==0: routing_status = agent.get_path(g=network.g)
            agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
            # reroute upon closure
            if agent.next_link in closed_links:
                routing_status = agent.get_path(g=network.g)
                agent.find_next_link(t, node2link_dict=network.node2link_dict)
        t_agent_1 = time.time()

        ### link model
        ### Each iteration in the link model is not time-consuming. So just keep using one process.
        t_link_0 = time.time()
        network, burnt_links, closed_links = link_model(t, network, link_fire_time_dict, burnt_links, closed_links)
        t_link_1 = time.time()
        
        ### node model
        t_node_0 = time.time()
        network, move, check_traffic_flow_links_dict = node_model(t, network, move, check_traffic_flow_links_dict)
        t_node_1 = time.time()

        ### metrics
        if t%100 == 0:
            arrival_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='arrive'])
            shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_arrive'])
            if len(network.agents)==0:
                logging.info("all agents arrive at destinations")
                break
            veh_loc = [network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]].midpoint for agent in network.agents.values()]
            # outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
            outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = 0, 0
            avg_fire_dist = 0 # np.mean(fire_point_distance(veh_loc))
            neg_dist = cnt_veh_in_fire(t, veh_loc=veh_loc, fire_df=fire_df)
            ### arrival
            with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
                t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, shelter_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")
            ### transfer
            with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
                transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")

            ### fitness metric
            fitness += neg_dist
        
        ### stepped outputs
        if t%1200==0:
            link_output = pd.DataFrame([(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2)) for link in network.links.values() if link.link_type=='real'], columns=['link_id', 'q', 'r', 't'])
            link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
            node_agent_cnts = pd.DataFrame([(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
            node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

        if t%100==0: 
            logging.info(" ".join([str(i) for i in [t, arrival_cnts, shelter_cnts, move, '|', len(burnt_links), len(closed_links), '|', round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))

    return fitness


### python3 -c 'import dta_meso_butte; dta_meso_butte.main(random_seed=0, dept_time_id="imm", reroute_pct=0, phase_tdiff=0, counterflow=0)'
if __name__ == "__main__":
    fire_df = pd.read_csv("projects/butte_osmnx/demand_inputs/simulation_fire_locations.csv")
    fire_df['start_time'] = np.where(np.isnan(fire_df['start_time']), np.random.randint(0, 7200, fire_df.shape[0]), fire_df['start_time'])
    fire_df['end_time'] = np.where(np.isnan(fire_df['end_time']), fire_df['start_time'], fire_df['end_time'])
    # print(fire_df.head())
    print(fire_df.tail())
    main(random_seed=0, fire_df=fire_df, dept_time_col='dept_time_scen_1')
