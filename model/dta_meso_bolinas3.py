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
scratch_dir = '/home/bingyu/Documents/spatial_queue/projects/bolinas/simulation_outputs' # os.environ['OUTPUT_FOLDER']
### user
sys.path.insert(0, home_dir)
import util.haversine as haversine
from model.queue_class import Network, Node, Link, Agent

import warnings
warnings.filterwarnings('error', message='Creating an ndarray from ragged*')

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
    evacuation_zone_dist = haversine.scipy_point_to_vertex_dist(vehicle_array, evacuation_zone)
    evacuation_buffer_dist = haversine.scipy_point_to_vertex_dist(vehicle_array, evacuation_buffer)
    return np.sum(evacuation_zone_dist>0), np.sum(evacuation_buffer_dist>0)
    

def link_model(t, network, links_raster, link_closed_time=None, closed_mode=None, fire_array=None, flame_array=None, eucalyptus_array=None):
    # t=7920
    links_in_fire = []
    closed_links_in_fire = []
    for links_raster_oneside in links_raster:
        # links in fire
        links_in_fire_oneside = np.where((fire_array>=t) & (fire_array<t+1), links_raster_oneside, np.nan)
        links_in_fire_oneside = links_in_fire_oneside[~np.isnan(links_in_fire_oneside)]
        links_in_fire += np.unique(links_in_fire_oneside).tolist()
        # some are closed
        if closed_mode == 'flame':
            closed_links_in_fire_oneside = np.where((fire_array>=t) & (fire_array<t+1) & (flame_array>2), links_raster_oneside, np.nan)
        # elif closed_mode == 'tree':
        #     closed_links_in_fire_oneside = np.where((eucalyptus_array==1), links_raster_oneside, np.nan)
        else:
            closed_links_in_fire_oneside = []
        # closed_links_in_fire_oneside = closed_links_in_fire_oneside[~np.isnan(closed_links_in_fire_oneside)]
        closed_links_in_fire += np.unique(closed_links_in_fire_oneside).tolist()
    links_in_fire = set(links_in_fire)
    closed_links_in_fire = set(closed_links_in_fire)
    # print(closed_links_in_fire)
    ### raster fire intersection: only close residential and unclassified roads. no harm to virtual links or higher class links.
    for link_id in links_in_fire:
        # this link
        link = network.links[link_id]
        # burning links
        if (link.status == 'open'): 
            link.status = 'burning'
            link.burnt_time = t

    for link_id in closed_links_in_fire:
        link = network.links[link_id]
        if link.closed_time == 3600*100:
            link.close_link_to_newcomers(g=network.g)
            link.status = 'burning_closed'
            link.closed_time = t

    if (closed_mode == 'tree'):
        for link_id in [543, 544, 601, 602]:
            link = network.links[link_id]
            if t<3600*link_closed_time:
                link.close_link_to_newcomers(g=network.g)
                link.status = 'closed'
                # print('close {}'.format(link_id))
            elif t==3600*link_closed_time:
                link.open_link_to_newcomers(g=network.g)
                link.status = 'open'
            else:
                pass
    
    # run link model; close and open links due to fire
    for link_id, link in network.links.items(): 
        ### link status: open (initial); closed (forever closed); burning (on fire but not closed); burning_closed (on fire and closed); burnt_over (fire moved over and will not be on fire again         )
        link.run_link_model(t, agent_id_dict=network.agents)
        
        ### reopen some links
        if (link.status in ['burning']) and (t-link.burnt_time>=3600):
            link.status='burnt_over'
        if (link.status in ['burning_closed']) and (t-link.closed_time>=3600*link_closed_time):
            link.open_link_to_newcomers(g=network.g)
            if (t-link.burnt_time>=3600):
                link.status='burnt_over'
            else:
                link.status='burning'
        # if (link_id==543) and (t%1200==0): print(t, link.status, link.queue_vehicles, link.total_vehicles_left)

    return network

def node_model(t, network, move, check_traffic_flow_links_dict, special_nodes=None):
    # only run node model for those with vehicles waiting
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles)>0])

    # run node model
    for node_id in node_ids_to_run:
        node = network.nodes[node_id] 
        n_t_move, transfer_links= node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links, agent_id_dict=network.agents, node2link_dict=network.node2link_dict, special_nodes=special_nodes)
        move += n_t_move
        ### how many people moves across a specified link pair in this step
        for transfer_link in transfer_links:
            if transfer_link in check_traffic_flow_links_dict.keys():
                check_traffic_flow_links_dict[transfer_link] += 1
        
    return network, move, check_traffic_flow_links_dict

def one_step(t, data, config, update_data):
    # print(t)

    network, evacuation_zone, evacuation_buffer, links_raster, fire_array, flame_array, eucalyptus_array = data['network'], data['evacuation_zone'], data['evacuation_buffer'], data['links_raster'], data['fire_array'], data['flame_array'], data['eucalyptus_array']
    
    scen_nm, simulation_outputs, check_traffic_flow_links, fire_id, comm_id, special_nodes, link_closed_time, closed_mode, shelter_scen_id = config['scen_nm'], config['simulation_outputs'], config['check_traffic_flow_links'], config['fire_id'], config['comm_id'], config['special_nodes'], config['link_closed_time'], config['closed_mode'], config['shelter_scen_id']

    in_fire_dict, shelter_capacity_122, shelter_capacity_202 = update_data['in_fire_dict'], update_data['shelter_capacity_122'], update_data['shelter_capacity_202']

    # link closure probability when fire arrives
    # clos_id_dict = {'1': 0, '2': 0.15, '3': 1}
    # link_closure_prob = clos_id_dict[clos_id]
    # link_closure_prob = 1

    move = 0
    congested = 0
    step_fitness = None
    
    ### update link travel time before rerouting
    reroute_freq_dict = {'1': 300, '2': 900, '3': 1800}
    reroute_freq = reroute_freq_dict[comm_id]
    # if (t%reroute_freq==0):
    #     for link in network.links.values():
    #         link.update_travel_time_by_queue_length(network.g)
    # reroute_freq = 300
    
    ### reset link congested counter
    for link in network.links.values():
        link.congested = 0
        if (t%100==0):
            link.update_travel_time_by_queue_length(network.g)

    # vehicle locations
    veh_loc = [network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]].midpoint for agent in network.agents.values()]
    veh_loc_id = [agent_id for agent_id in network.agents.keys()]
    fire_loc = []
    # in_fire_list = []
    for link in network.links.values():
        if link.status in ['burning_closed', 'burning']:
            fire_loc.append(link.midpoint)
            for agent_id in link.run_vehicles + link.queue_vehicles:
                try: in_fire_dict[agent_id][1] += 1
                except KeyError:
                    in_fire_dict[agent_id] = [network.agents[agent_id].agent_type, 1]
    # temporarily safe
    outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
    # distance to fire
    if len(fire_loc) == 0:
        vehicle_fire_distance = np.ones(len(veh_loc_id))*100000
    else:
        vehicle_fire_distance = haversine.scipy_point_to_vertex_distance_positive(np.array(veh_loc), np.array(fire_loc))
    avg_fire_dist = np.mean(vehicle_fire_distance)
    # vehicles close to fire go
    see_fire_veh_loc_id = [agent_id for agent_id in np.array(veh_loc_id)[vehicle_fire_distance<=400] if network.agents[agent_id].status=='unloaded']
    # print('cognitive', t, len(see_fire_veh_loc_id))
    for agent_id in see_fire_veh_loc_id:
        network.agents[agent_id].departure_time = t
    if t%3600==0:
        print(t)

    ### agent model
    t_agent_0 = time.time()
    stopped_agents_list = []
    for agent_id, agent in network.agents.items():
        ### first remove arrived vehicles
        if agent.status == 'arrive':
            network.agents_stopped[agent_id] = (agent.status, t, agent.agent_type)
            stopped_agents_list.append(agent_id)
            continue
        ### find congested vehicles: spent too long in a link
        current_link = network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]]
        if (current_link.link_type != 'v') and (agent.current_link_enter_time is not None) and (t-agent.current_link_enter_time>3600*0.5):
            congested += 1
            current_link.congested += 1
            # print(shelter_scen_id, type(shelter_scen_id))
            # sys.exit(0)
            if (shelter_scen_id=='0'):
                if (t-agent.current_link_enter_time>3600*3):# or ((t-agent.current_link_enter_time>3600*0.5) and (current_link.status in ['burning', 'burning_closed']) and (t-current_link.burnt_time>3600*0.5)):
                    agent.status = 'shelter_a1'
            elif shelter_scen_id=='1':
                pass
            else:
                pass
        if (shelter_scen_id=='2'):
            if (agent.current_link_end_nid==122) and (shelter_capacity_122>0):
                agent.status = 'shelter_park'
                shelter_capacity_122 -= 1
            elif (agent.current_link_end_nid==202) and (shelter_capacity_202>0):
                agent.status = 'shelter_park'
                shelter_capacity_202 -= 1
            else:
                pass
            print('shelter_capacity {}, {}'.format(shelter_capacity_122, shelter_capacity_202))
        ### agents need rerouting
        # initial route 
        if (t==0) or (t%reroute_freq==agent_id%reroute_freq):
            routing_status = agent.get_path(t, g=network.g)
            agent.find_next_link(node2link_dict=network.node2link_dict)
        agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
        ### remove passively sheltered vehicles immediately, no need to wait for node model
        if agent.status in ['shelter_p', 'shelter_a1', 'shelter_park']:
            current_link.queue_vehicles = [v for v in current_link.queue_vehicles if v!=agent_id]
            current_link.run_vehicles = [v for v in current_link.run_vehicles if v!=agent_id]
            network.nodes[agent.current_link_end_nid].shelter_counts += 1
            network.agents_stopped[agent_id] = (agent.status, t, agent.agent_type)
            stopped_agents_list.append(agent_id)
    for agent_id in stopped_agents_list:
        del network.agents[agent_id]
    t_agent_1 = time.time()

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network, links_raster, link_closed_time=link_closed_time, closed_mode=closed_mode, fire_array=fire_array, flame_array=flame_array, eucalyptus_array=eucalyptus_array)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    network, move, check_traffic_flow_links_dict = node_model(t, network, move, check_traffic_flow_links_dict, special_nodes=special_nodes)
    t_node_1 = time.time()

    ### metrics
    if t%120 == 0:
        metrics = {'local': {'unloaded': 0, 'enroute': 0, 'arrive': 0, 'shelter_a1': 0, 'shelter_park': 0, 'shelter_p': 0, 'in_fire_cnts': 0, 'in_fire_time': 0}, 'visitor': {'unloaded': 0, 'enroute': 0, 'arrive': 0, 'shelter_a1': 0, 'shelter_park': 0, 'shelter_p': 0, 'in_fire_cnts': 0, 'in_fire_time': 0}}
        for agent_id, agent_info in network.agents.items():
            if agent_info.agent_type == 'local':
                if agent_info.status == 'unloaded': metrics['local']['unloaded'] += 1
                elif agent_info.status == 'enroute': metrics['local']['enroute'] += 1
                else: pass
            elif agent_info.agent_type == 'visitor':
                if agent_info.status == 'unloaded': metrics['visitor']['unloaded'] += 1
                elif agent_info.status == 'enroute': metrics['visitor']['enroute'] += 1
                else: pass
            else: pass
        for agent_id, agent_info in network.agents_stopped.items():
            if agent_info[2] == 'local':
                if agent_info[0] == 'arrive': metrics['local']['arrive'] += 1
                elif agent_info[0] == 'shelter_a1': metrics['local']['shelter_a1'] += 1
                elif agent_info[0] == 'shelter_park': metrics['local']['shelter_park'] += 1
                elif agent_info[0] == 'shelter_p': metrics['local']['shelter_p'] += 1
                else: pass
            elif agent_info[2] == 'visitor':
                if agent_info[0] == 'arrive': metrics['visitor']['arrive'] += 1
                elif agent_info[0] == 'shelter_a1': metrics['visitor']['shelter_a1'] += 1
                elif agent_info[0] == 'shelter_park': metrics['visitor']['shelter_park'] += 1
                elif agent_info[0] == 'shelter_p': metrics['visitor']['shelter_p'] += 1
                else: pass 
        for agent_id, [agent_type, agent_in_fire_time] in in_fire_dict.items():
            if agent_type == 'local':
                metrics['local']['in_fire_cnts'] += 1
                metrics['local']['in_fire_time'] += agent_in_fire_time
            elif agent_type == 'visitor':
                metrics['visitor']['in_fire_cnts'] += 1
                metrics['visitor']['in_fire_time'] += agent_in_fire_time
            else: pass
        burning_links = [link_id for link_id, link in network.links.items() if link.status=='burning']
        burning_closed_links = [link_id for link_id, link in network.links.items() if link.status=='burning_closed']
        burnt_over_links = [link_id for link_id, link in network.links.items() if link.status=='burnt_over']
        closed_links = [link_id for link_id, link in network.links.items() if link.status=='closed']
        total_run = sum([len(link.run_vehicles) for link in network.links.values() if link.link_type!='v'])
        total_queue = sum([len(link.queue_vehicles) for link in network.links.values() if link.link_type!='v'])
        # logging
        # print('metrics', metrics['local']['shelter_a1'], metrics['visitor']['shelter_a1'])
        logging.info(" ".join([str(i) for i in [t, '|',
            metrics['local']['unloaded'], metrics['local']['enroute'], metrics['local']['arrive'], metrics['local']['shelter_a1'], metrics['local']['shelter_park'], metrics['local']['shelter_p'], metrics['local']['in_fire_cnts'], metrics['local']['in_fire_time'], '|',
            metrics['visitor']['unloaded'], metrics['visitor']['enroute'], metrics['visitor']['arrive'], metrics['visitor']['shelter_a1'], metrics['visitor']['shelter_park'], metrics['visitor']['shelter_p'], metrics['visitor']['in_fire_cnts'], metrics['visitor']['in_fire_time'], '|',
            move, congested, total_run, total_queue, len(veh_loc), '|', round(avg_fire_dist,2), outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, '|', len(burning_links), len(burning_closed_links), len(burnt_over_links), len(closed_links)]]))
        ### t_stats
        with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'a') as t_stats_outfile:
            t_stats_outfile.write(",".join([str(x) for x in [t, 
            metrics['local']['unloaded'], metrics['local']['enroute'], metrics['local']['arrive'], metrics['local']['shelter_a1'], metrics['local']['shelter_park'], metrics['local']['shelter_p'], metrics['local']['in_fire_cnts'], metrics['local']['in_fire_time'],
            metrics['visitor']['unloaded'], metrics['visitor']['enroute'], metrics['visitor']['arrive'], metrics['visitor']['shelter_a1'], metrics['visitor']['shelter_park'], metrics['visitor']['shelter_p'], metrics['visitor']['in_fire_cnts'], metrics['visitor']['in_fire_time'],
            move, congested, total_run, total_queue, len(veh_loc), round(avg_fire_dist,2), outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, len(burning_links), len(burning_closed_links), len(burnt_over_links), len(closed_links)]]) + "\n")
        ### transfer
        # with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
        #     transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")
    
    ### stepped outputs
    if t%120==0:
        link_output = pd.DataFrame(
            [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2), link.total_vehicles_left, link.congested) for link in network.links.values() if (link.link_type!='v') and (len(link.queue_vehicles)+len(link.run_vehicles)+link.total_vehicles_left+link.congested>0)], columns=['link_id', 'q', 'r', 't', 'tot_left', 'congested'])
        link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)  
        ### node agent cnts
        node_agent_cnts = pd.DataFrame(
            [(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
        node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)
        ### node move cnts
        node_move_cnts = pd.DataFrame([(node.node_id, node.node_move, node.shelter_counts, node.x, node.y) for node in network.nodes.values() if (node.node_type!='v') and (node.node_move + node.shelter_counts>0)], columns=['node_id', 'node_move', 'shelter_counts', 'x', 'y'])
        node_move_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_move_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)
        
    # stop
    if len(network.agents)==0:
        logging.info("all agents arrive at destinations")
        return step_fitness, network, 'stop', {'in_fire_dict': in_fire_dict, 'shelter_capacity_122': shelter_capacity_122, 'shelter_capacity_202': shelter_capacity_202}
    else:
        return step_fitness, network, 'continue', {'in_fire_dict': in_fire_dict, 'shelter_capacity_122': shelter_capacity_122, 'shelter_capacity_202': shelter_capacity_202}

def preparation(random_seed=0, fire_id=None, comm_id=None, vphh=None, visitor_cnts=None, contra_id=None, link_closed_time=None, closed_mode=None, shelter_scen_id=None, scen_nm=None):
    ### logging and global variables

    project_location = '/projects/bolinas'
    network_file_edges = project_location + '/network_inputs/bolinas_edges_sim.csv'
    network_file_nodes = project_location + '/network_inputs/bolinas_nodes_sim.csv'
    network_file_special_nodes = project_location + '/network_inputs/bolinas_special_nodes.json'
    network_file_edges_raster1 = project_location + '/network_inputs/bolinas_edges_raster1.tif'
    network_file_edges_raster2 = project_location + '/network_inputs/bolinas_edges_raster2.tif'
    demand_files = [project_location + '/demand_inputs/od_csv/resident_visitor_od_rs{}_commscen{}_vphh{}_visitor{}.csv'.format(random_seed, comm_id, vphh, visitor_cnts)]
    simulation_outputs = '' ### scratch_folder
   
    if contra_id=='0': cf_files = []
    elif contra_id=='1': cf_files = [project_location + '/network_inputs/bolinas_contraflow.csv']
    else: cf_files = []

    scen_nm = scen_nm
    logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)
    print('log file created for {}'.format(scen_nm))

    ### network
    links_raster = [rio.open(work_dir + network_file_edges_raster1).read(1), rio.open(work_dir + network_file_edges_raster2).read(1)]
    with open(work_dir + network_file_special_nodes) as special_nodes_file:
        special_nodes = json.load(special_nodes_file)
    network = Network()
    network.dataframe_to_network(project_location=project_location, network_file_edges = network_file_edges, network_file_nodes = network_file_nodes, cf_files = cf_files, special_nodes=special_nodes, scen_nm=scen_nm)
    network.add_connectivity()

    ### demand
    network.add_demand(demand_files = demand_files)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    ### evacuation zone
    evacuation_zone_df = pd.read_csv(work_dir + project_location + '/network_inputs/bolinas_boundary.csv')
    evacuation_zone_gdf = gpd.GeoDataFrame(evacuation_zone_df, crs='epsg:4326', geometry = evacuation_zone_df['WKT'].map(loads)).to_crs('epsg:26910')
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf['geometry'].buffer(1609).unary_union
    logging.info('Evacuation zone is {:.2f} km2, considering 1 mile buffer it is {:.2f} km2'.format(evacuation_zone.area/1e6, evacuation_buffer.area/1e6))
    
    ### fire
    fire_array = rio.open(work_dir + project_location + '/demand_inputs/flamelength/time_fire{}_match_road.tif'.format(fire_id)).read(1)*3600
    flame_array = rio.open(work_dir + project_location + '/demand_inputs/flamelength/flame_fire{}_match_road.tif'.format(fire_id)).read(1)
    eucalyptus_array = rio.open(work_dir + project_location + '/demand_inputs/fire/eucalyptus_match_roads.tif').read(1)
    # print(np.sum([fire_array<3600]))
    # print(np.min(fire_array))
    # sys.exit(0)

    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t',
        'local_unloaded', 'local_enroute', 'local_arr', 'local_shelter_a1', 'local_shelter_park', 'local_shelter_p', 'local_in_fire_cnts', 'local_in_fire_times',
        'visitor_unloaded', 'visitor_enroute', 'visitor_arr', 'visitor_shelter_a1', 'visitor_shelter_park', 'visitor_shelter_p', 'visitor_in_fire_cnts', 'visitor_in_fire_times',
        'move', 'congested', 'total_run', 'total_queue', 'veh_loc_length', 'avg_fdist', 'out_evac_zone_cnts', 'out_evac_buffer_cnts', 'burning_links', 'burning_closed_links', 'burnt_over_links', 'closed_links'])+"\n")
    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = []
    # with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(
    #     scen_nm), 'w') as transfer_stats_outfile:
    #     transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    return {'network': network, 'evacuation_zone': evacuation_zone, 'evacuation_buffer': evacuation_buffer, 'links_raster': links_raster, 'fire_array': fire_array, 'flame_array': flame_array, 'eucalyptus_array': eucalyptus_array}, {'check_traffic_flow_links': check_traffic_flow_links, 'scen_nm': scen_nm, 'simulation_outputs': simulation_outputs, 'fire_id': fire_id, 'comm_id': comm_id, 'special_nodes': special_nodes, 'link_closed_time': link_closed_time, 'closed_mode': closed_mode, 'shelter_scen_id': shelter_scen_id}, {'in_fire_dict': {}, 'shelter_capacity_122': 200, 'shelter_capacity_202': 100}

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
