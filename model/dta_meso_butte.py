#!/usr/bin/env python
# coding: utf-8
import os
import gc
import sys
import time
import json 
import random
import logging 
import warnings
import numpy as np
import pandas as pd 
import rasterio as rio
import scipy.io as sio
import geopandas as gpd
from ctypes import c_double
from shapely.wkt import loads
from shapely.geometry import Point
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

### numbers of vehicles that have left the evacuation zone / buffer distance
def outside_polygon(evacuation_zone, evacuation_buffer, vehicle_locations):
    vehicle_array = np.array(vehicle_locations)
    evacuation_zone_dist = haversine.point_to_vertex_dist(vehicle_array, evacuation_zone)
    evacuation_buffer_dist = haversine.point_to_vertex_dist(vehicle_array, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)

def link_model(t, network, links_raster, reroute_freq, link_closure_prob=None):
    if (t%300==0) and (t<=340*60):
        fire_array = rio.open(work_dir + '/projects/butte_osmnx/demand_inputs/fire_cawfe/cawfe_spot_fire_{}.tif'.format(int(t//60)+105)).read(1)
        links_in_fire = np.where(fire_array==0, 0, links_raster)
        links_in_fire = np.unique([links_in_fire.tolist()])
        print('# of links in fire: {}'.format(len(links_in_fire)))
    else:
        links_in_fire = []

    # force to close these links at given time
    if t == 1800: ### pentz rd
        network.links[24838].close_link_to_newcomers(g=network.g)
        network.links[24838].burnt = 'burnt'
        network.links[24875].close_link_to_newcomers(g=network.g)
        network.links[24875].burnt = 'burnt'
    elif t == 3600: ### pentz rd
        network.links[3425].close_link_to_newcomers(g=network.g)
        network.links[3425].burnt = 'burnt'
        network.links[20912].close_link_to_newcomers(g=network.g)
        network.links[20912].burnt = 'burnt'
    elif t == 10800: ### pentz rd
        network.links[9012].close_link_to_newcomers(g=network.g)
        network.links[9012].burnt = 'burnt'
        network.links[14363].close_link_to_newcomers(g=network.g)
        network.links[14363].burnt = 'burnt'
    else:
        pass
    
    for link_id, link in network.links.items(): 
        link.run_link_model(t, agent_id_dict=network.agents)
        if link.link_type == 'v': ### do not track the link time of virtual links
            link.travel_time_list = []
            pass
        else:
            link.travel_time_list = []
            if (t+1)%reroute_freq==0: link.update_travel_time_by_queue_length(network.g, len(link.queue_vehicles))

        # raster fire intersection
        if (link.link_type in ['residential', 'unclassified']) and (link.link_id in links_in_fire) and (link.burnt == 'not_yet'):
            if np.random.uniform(0,1) < link_closure_prob:
                link.close_link_to_newcomers(g=network.g)
                # print(link.link_id)
            link.burnt = 'burnt'
        else: pass
    return network

def node_model(t, network, move, check_traffic_flow_links_dict, special_nodes=None):
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles)>0])
    for node_id in node_ids_to_run:
        node = network.nodes[node_id] 
        n_t_move, transfer_links, agent_update_dict, link_update_dict = node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links, agent_id_dict=network.agents, node2link_dict=network.node2link_dict, special_nodes=special_nodes)
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
                agent.find_next_link(node2link_dict = network.node2link_dict)

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

def one_step(t, data, config):

    network, links_raster, evacuation_zone, evacuation_buffer, fire_df = data['network'], data['links_raster'], data['evacuation_zone'], data['evacuation_buffer'], data['fire_df']
    
    scen_nm, simulation_outputs, rout_id, check_traffic_flow_links, reroute_freq, clos_id, special_nodes = config['scen_nm'], config['simulation_outputs'], config['rout_id'], config['check_traffic_flow_links'], config['reroute_freq'], config['clos_id'], config['special_nodes']

    # link closure probability when fire arrives
    clos_id_dict = {'1': 0, '2': 0.1, '3': 1}
    link_closure_prob = clos_id_dict[clos_id]

    move = 0
    step_fitness = None
    ### agent model
    t_agent_0 = time.time()
    for agent_id, agent in network.agents.items():
        if rout_id == '1':
            # initial route 
            if (t==0) or (t%reroute_freq==0) and (agent.status != 'shelter'): routing_status = agent.get_path(g=network.g)
            agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
            # reroute upon closure
            if (agent.next_link is not None) and (network.links[agent.next_link].status=='closed'):
                routing_status = agent.get_path(g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
        if rout_id == '2':
            # initial route 
            if (t==0) or (t%reroute_freq==0 and t < 9000) and (agent.status != 'shelter'):
                routing_status = agent.get_path(g=network.g)
                if agent.agent_id == 7910: print(t, agent.current_link_end_nid, agent.route)
            agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
            # reroute upon closure
            if (t<9000) and (agent.next_link is not None) and (network.links[agent.next_link].status=='closed'):
                routing_status = agent.get_path(g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
            elif (t>=9000) and (agent.next_link is not None) and (network.links[agent.next_link].status=='closed'):
                routing_status = agent.get_partial_path(g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
            else:
                pass
    t_agent_1 = time.time()

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network, links_raster, reroute_freq, link_closure_prob=link_closure_prob)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    network, move, check_traffic_flow_links_dict = node_model(t, network, move, check_traffic_flow_links_dict, special_nodes=special_nodes)
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
        neg_dist = 0 # count_vehicles_in_fire(t, vehicle_locations=veh_loc, fire_df=fire_df)
        ### arrival
        with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
            t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, shelter_cnts, move, round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")
        ### transfer
        with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
            transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")

        ### fitness metric
        step_fitness = neg_dist
    
    ### stepped outputs
    if t%120==0:
        link_output = pd.DataFrame(
            [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2)) for link in network.links.values() if (link.link_type=='real') and (len(link.queue_vehicles)+len(link.run_vehicles)>0)], columns=['link_id', 'q', 'r', 't'])
        link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
    if t%1200==0:    
        node_agent_cnts = pd.DataFrame(
            [(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
        node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

    if t%100==0: 
        burnt_links = [link_id for link_id, link in network.links.items() if link.burnt=='burnt']
        closed_links = [link_id for link_id, link in network.links.items() if link.status=='closed']
        logging.info(" ".join([str(i) for i in [t, arrival_cnts, shelter_cnts, move, '|', len(burnt_links), len(closed_links), '|', round(avg_fire_dist,2), neg_dist, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))
    return step_fitness, network

def preparation(random_seed=None, vphh_id='123', dept_id='2', clos_id='2', contra_id='0', rout_id='2', scen_nm=None):
    ### logging and global variables

    network_file_edges = '/projects/butte_osmnx/network_inputs/butte_edges_sim.csv'
    network_file_nodes = '/projects/butte_osmnx/network_inputs/butte_nodes_sim.csv'
    network_file_special_nodes = '/projects/butte_osmnx/network_inputs/butte_special_nodes.json'
    network_file_edges_raster = '/projects/butte_osmnx/network_inputs/butte_edges_sim.tif'
    demand_files = ["/projects/butte_osmnx/demand_inputs/od_vphh{}_dept{}.csv".format(vphh_id, dept_id)]
    simulation_outputs = '' ### scratch_folder
    if contra_id=='0': cf_files = []
    elif contra_id=='3': cf_files = ['/projects/butte_osmnx/network_inputs/contraflow_skyway_3.csv']
    elif contra_id=='4': cf_files = ['/projects/butte_osmnx/network_inputs/contraflow_skyway_4.csv']
    else: cf_files = []
    reroute_freq = 300

    scen_nm = scen_nm
    print('scenario ', scen_nm)
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)
    print('log file created')

    ### network
    links_raster = rio.open(work_dir + network_file_edges_raster).read(1)
    special_nodes = json.load(open(work_dir + network_file_special_nodes))
    network = Network()
    network.dataframe_to_network(network_file_edges = network_file_edges, network_file_nodes = network_file_nodes, cf_files = cf_files, special_nodes=special_nodes)
    network.add_connectivity()
    ### traffic signal
    # network.links[21671].capacity /= 2

    ### demand
    network.add_demand(demand_files = demand_files)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    ### evacuation zone
    evacuation_zone_gdf = gpd.read_file(work_dir+'/projects/butte_osmnx/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp').to_crs('epsg:26910')
    evacuation_zone_gdf = evacuation_zone_gdf.loc[evacuation_zone_gdf['id']<=14].copy()
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf['geometry'].buffer(1609).unary_union
    logging.info('Evacuation zone is {:.2f} km2, considering 1 mile buffer it is {:.2f} km2'.format(evacuation_zone.area/1e6, evacuation_buffer.area/1e6))

    ### process the fire information
    fire_df = pd.read_csv("projects/butte_osmnx/demand_inputs/simulation_fire_locations.csv")
    fire_df = fire_df[fire_df['type'].isin(['pentz', 'clark', 'neal'])]
    ### fire arrival time
    ### additional fire spread information will be considered later
    
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'shelter', 'move', 'avg_fdist', 'neg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")
    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = [(29,33)]
    with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(
        scen_nm), 'w') as transfer_stats_outfile:
        transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    return {'network': network, 'links_raster': links_raster, 'evacuation_zone': evacuation_zone, 'evacuation_buffer': evacuation_buffer, 'fire_df': fire_df}, {'check_traffic_flow_links': check_traffic_flow_links, 'scen_nm': scen_nm, 'simulation_outputs': simulation_outputs, 'rout_id': rout_id, 'clos_id': clos_id, 'reroute_freq': reroute_freq, 'special_nodes': special_nodes}

# def main(random_seed=None, dept_time_col=None):
    
#     fitness = 0
#     network, link_fire_time_dict, check_traffic_flow_links_dict, burnt_links, closed_links = preparation(
#         random_seed=random_seed, dept_time_col=dept_time_col)

#     t_s, t_e = 0, 18001
#     for t in range(t_s, t_e):
#         step_fitness = one_step(t, network, link_fire_time_dict, check_traffic_flow_links_dict, burnt_links, closed_links)
#         fitness += step_fitness
    
#     return fitness

### python3 -c 'import dta_meso_butte; dta_meso_butte.main(random_seed=0, dept_time_id="imm", reroute_pct=0, phase_tdiff=0, counterflow=0)'
# if __name__ == "__main__":
#     main(random_seed=0, dept_time_col='dept_time_scen_1')
