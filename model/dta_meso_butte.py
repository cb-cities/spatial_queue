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
import scipy.spatial.distance
### dir
home_dir = '/home/bingyu/Documents/spatial_queue' # os.environ['HOME']+'/spatial_queue'
work_dir = '/home/bingyu/Documents/spatial_queue' # os.environ['WORK']+'/spatial_queue'
scratch_dir = '/home/bingyu/Documents/spatial_queue/projects/butte_osmnx/simulation_outputs' # os.environ['OUTPUT_FOLDER']
### user
sys.path.insert(0, home_dir)
import util.haversine as haversine
from model.queue_class import Network, Node, Link, Agent

random.seed(0)
np.random.seed(0)

# warnings.filterwarnings("error")

def count_vehicles_in_fire(t_now, vehicle_locations=None, fire_df=None):
    # vehicle locations
    vehicle_array = np.array(vehicle_locations)
    # fire start locations
    fire_origin_array = fire_df[['x', 'y']].values
    # vehicle and fire start location matrix
    vehicle_fire_origin_distance_matrix = scipy.spatial.distance.cdist(vehicle_array, fire_origin_array)
    # vehicle and current fire boundary
    vehicle_fire_distance_matrix = vehicle_fire_origin_distance_matrix - fire_df['speed'].values * np.minimum(np.maximum(t_now, fire_df['start_time'].values), fire_df['end_time'].values)
    + fire_df['speed'].values * fire_df['start_time'].values
    # distance of vehicle to closest fire
    vehicle_fire_distance_array = np.min(vehicle_fire_distance_matrix, axis=1)
    return np.sum(vehicle_fire_distance_array <= 0)

def road_closure_time(link_id_dict, fire_gdf):
    ### https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
    link_geom_df = pd.DataFrame([[link_id, link.geometry] for link_id, link in link_id_dict.items() if link.link_type != 'v'], columns=['link_id', 'geometry'])
    link_geom_gdf = gpd.GeoDataFrame(link_geom_df, crs='epsg:26910', geometry=link_geom_df['geometry'])

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
    tmp["fire_arr"] = np.minimum(tmp['end_time'], np.maximum(tmp['fire_arr'], tmp['start_time']))

    tmp = tmp.sort_values(by='fire_arr', ascending=True).groupby('link_id').first().reset_index()

    for row in tmp.itertuples():
        link_id_dict[getattr(row, 'link_id')].burnt = 'not_yet'
        link_id_dict[getattr(row, 'link_id')].fire_type = getattr(row, 'type')
        link_id_dict[getattr(row, 'link_id')].fire_time = getattr(row, 'fire_arr')

### numbers of vehicles that have left the evacuation zone / buffer distance
def outside_polygon(evacuation_zone, evacuation_buffer, vehicle_locations):
    vehicle_array = np.array(vehicle_locations)
    evacuation_zone_dist = haversine.point_to_vertex_dist(vehicle_array, evacuation_zone)
    evacuation_buffer_dist = haversine.point_to_vertex_dist(vehicle_array, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)

def link_model(t, network):
    for link_id, link in network.links.items(): 
        link.run_link_model(t, agent_id_dict=network.agents)
        if link.link_type == 'v': ### do not track the link time of virtual links
            link.travel_time_list = []
        else:
            link.travel_time_list = []
        # link.update_travel_time(self, t_now, link_time_lookback_freq=None, g=None, update_graph=False)
        if link.fire_time == t:
            if (link.fire_type in ['ember', 'initial']) and (random.uniform(0, 1) < 0.1):
                link.close_link_to_newcomers(g=network.g)
                link.burnt = 'burnt'
            elif (link.fire_type in ['pentz', 'neal', 'clark']):
                link.close_link_to_newcomers(g=network.g)
                link.burnt = 'burnt'
            else:
                link.burnt = 'not_burnt'
    return network

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

def one_step(t, network, evacuation_zone, evacuation_buffer, fire_df, check_traffic_flow_links, scen_nm, simulation_outputs):
    move = 0
    step_fitness = None
    ### agent model
    t_agent_0 = time.time()
    for agent_id, agent in network.agents.items():
        # initial route 
        if t==0: routing_status = agent.get_path(g=network.g)
        agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
        # reroute upon closure
        if (agent.next_link is not None) and (network.links[agent.next_link].status=='closed'):
            routing_status = agent.get_path(g=network.g)
            agent.find_next_link(t, node2link_dict=network.node2link_dict)
    t_agent_1 = time.time()

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    network, move, check_traffic_flow_links_dict = node_model(t, network, move, check_traffic_flow_links_dict)
    t_node_1 = time.time()

    ### metrics
    if t%100 == 0:
        arrival_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='arrive'])
        shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_arrive'])
        if len(network.agents)==0:
            logging.info("all agents arrive at destinations")
            return 0
        # vehicle locations
        veh_loc = [network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]].midpoint for agent in network.agents.values()]
        # temporarily safe
        outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
        avg_fire_dist = 0 # np.mean(fire_point_distance(veh_loc))
        neg_dist = count_vehicles_in_fire(t, vehicle_locations=veh_loc, fire_df=fire_df)
        ### arrival
        with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
            t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, shelter_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")
        ### transfer
        with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
            transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")

        ### fitness metric
        step_fitness = neg_dist
    
    ### stepped outputs
    if t%1200==0:
        link_output = pd.DataFrame(
            [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2)) for link in network.links.values() if link.link_type=='real'], columns=['link_id', 'q', 'r', 't'])
        link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
        node_agent_cnts = pd.DataFrame(
            [(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
        node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

    if t%100==0: 
        burnt_links = [link_id for link_id, link in network.links.items() if link.burnt=='burnt']
        closed_links = [link_id for link_id, link in network.links.items() if link.status=='closed']
        logging.info(" ".join([str(i) for i in [t, arrival_cnts, shelter_cnts, move, '|', len(burnt_links), len(closed_links), '|', round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))
    return step_fitness, network

def preparation(random_seed=None, dept_time_col=None, contraflow=False):
    ### logging and global variables

    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = '/projects/butte_osmnx/network_inputs/butte_edges_sim.csv'
    network_file_nodes = '/projects/butte_osmnx/network_inputs/butte_nodes_sim.csv'
    demand_files = ["/projects/butte_osmnx/demand_inputs/od.csv"]
    simulation_outputs = '' ### scratch_folder
    if contraflow: cf_files = []
    else: cf_files = []

    scen_nm = "full_dict_c0.1_d1"
    print(scen_nm)
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)
    print('log file created')

    ### network
    network = Network()
    network.dataframe_to_network(network_file_edges = network_file_edges, network_file_nodes = network_file_nodes)
    network.add_connectivity()

    ### demand
    network.add_demand(dept_time_col=dept_time_col, demand_files = demand_files)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    ### evacuation zone
    evacuation_zone_gdf = gpd.read_file(work_dir+'/projects/butte_osmnx/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp').to_crs('epsg:26910')
    evacuation_zone_gdf = evacuation_zone_gdf.loc[evacuation_zone_gdf['id']<=14].copy()
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf['geometry'].buffer(1609).unary_union
    logging.info('Evacuation zone is {:.2f} km2, considering 1 mile buffer it is {:.2f} km2'.format(evacuation_zone.area/1e6, evacuation_buffer.area/1e6))

    ### process the fire information
    fire_df = pd.read_csv("projects/butte_osmnx/demand_inputs/simulation_fire_locations.csv")
    fire_df['start_time'] = np.where(np.isnan(fire_df['start_time']), np.random.randint(0, 7200, fire_df.shape[0]), fire_df['start_time'])
    fire_df['end_time'] = np.where(np.isnan(fire_df['end_time']), fire_df['start_time'], fire_df['end_time'])
    fire_df = gpd.GeoDataFrame(fire_df, crs='epsg:4326', geometry=[Point(xy) for xy in zip(fire_df.lon, fire_df.lat)]).to_crs('epsg:26910')
    fire_df['x'] = fire_df['geometry'].apply(lambda x: x.x)
    fire_df['y'] = fire_df['geometry'].apply(lambda x: x.y)
    fire_df.to_csv(scratch_dir + simulation_outputs + '/fire/fire_init_{}.csv'.format(scen_nm), index=False)

    ### fire arrival time
    road_closure_time(network.links, fire_df)
    # pd.DataFrame([[link_id, link_fire_dist[0], link_fire_dist[1], link_fire_dist[2], network.links[link_id].geometry] for link_id, link_fire_dist in link_fire_time_dict.items()], columns=['link_id', 'link_fire_dist', 'fire_arrival_time', 'fire_type', 'geometry']).to_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm), index=False)
    # # sys.exit(0)
    # link_fire_time_df = pd.read_csv(scratch_dir + simulation_outputs + '/fire/road_fire_dist_{}.csv'.format(scen_nm))
    # link_fire_time_dict = {}
    # for row in link_fire_time_df.itertuples():
    #     link_id = getattr(row, 'link_id')
    #     link_fire_time_dict[link_id] = [getattr(row, 'link_fire_dist'), getattr(row, 'fire_arrival_time'), getattr(row, 'fire_type')]
    # print(link_fire_time_dict[link_id])
    
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'shelter', 'move', 'avg_fdist', 'neg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")
    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = [(29,33)]
    with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(
        scen_nm), 'w') as transfer_stats_outfile:
        transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    return network, evacuation_zone, evacuation_buffer, fire_df, check_traffic_flow_links, scen_nm, simulation_outputs

# def main(random_seed=None, dept_time_col=None):
    
#     fitness = 0
#     network, link_fire_time_dict, check_traffic_flow_links_dict, burnt_links, closed_links = preparation(
#         random_seed=random_seed, dept_time_col=dept_time_col)

#     t_s, t_e = 0, 18001
#     for t in range(t_s, t_e):
#         step_fitness = one_step(t, network, link_fire_time_dict, check_traffic_flow_links_dict, burnt_links, closed_links)
#         fitness += step_fitness
    
#     return fitness

def test(contraflow=False):
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    network_file_edges = 'projects/butte_osmnx/network_inputs/butte_edges_sim.csv'
    network_file_nodes = 'projects/butte_osmnx/network_inputs/butte_nodes_sim.csv'
    demand_files = ["projects/butte_osmnx/demand_inputs/od.csv"]
    simulation_outputs = '' ### scratch_folder
    if contraflow: cf_files = []
    else: cf_files = []

    scen_nm = "full_dict_c0.1_d1"
    print(scen_nm)
    print('test')

### python3 -c 'import dta_meso_butte; dta_meso_butte.main(random_seed=0, dept_time_id="imm", reroute_pct=0, phase_tdiff=0, counterflow=0)'
# if __name__ == "__main__":

#     main(random_seed=0, dept_time_col='dept_time_scen_1')
