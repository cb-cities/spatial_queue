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
work_dir = '/home/bingyu/Documents/traffic_data/butte_osmnx' # os.environ['WORK']+'/spatial_queue'
scratch_dir = '/home/bingyu/Documents/traffic_data/butte_osmnx/simulation_outputs' # os.environ['OUTPUT_FOLDER']
### user
sys.path.insert(0, home_dir)
import util.haversine as haversine
from model.queue_class import Network, Node, Link, Agent

random.seed(5)
np.random.seed(5)

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
    

def link_model(t, network, links_raster, reroute_freq, link_closure_prob=None, fire_array=None):
    if (t%300==0) and (t<=340*60):
        links_in_fire = np.where(fire_array!=(t//60)+105, 0, links_raster) ### --> should be np.nan?
        links_in_fire = np.unique(links_in_fire).tolist()
        ### raster fire intersection: only close residential and unclassified roads. no harm to virtual links or higher class links.
        for link_id in links_in_fire:
            # this link
            link = network.links[link_id]
            # opposite link
            try:
                opposite_link = network.links[network.node2link_dict[(link.end_nid, link.start_nid)]]
            except KeyError:
                opposite_link = None
            # burning links
            if (link.status == 'open') and (link.link_type in ['residential', 'unclassified']):
                link.status = 'burning'
                link.burnt_time = t
                if opposite_link is not None:
                    opposite_link.status = 'burning'
                    opposite_link.burnt_time = t
                if (np.random.uniform(0,1) < link_closure_prob):
                    link.close_link_to_newcomers(g=network.g)
                    link.status = 'burning_closed'
                    if opposite_link is not None:
                        opposite_link.close_link_to_newcomers(g=network.g)
                        opposite_link.status = 'burning_closed'
        print('# of links in fire: {}, at {}, examples {}'.format(len(links_in_fire), t, links_in_fire))
        # print('# of links in fire: {}'.format(len(links_in_fire)))
    else:
        links_in_fire = []
        
    # force to close these links at given time
    if t == 1800: ### pentz rd
        network.links[24838].close_link_to_newcomers(g=network.g)
        network.links[24838].status = 'closed'
        network.links[24875].close_link_to_newcomers(g=network.g)
        network.links[24875].status = 'closed'
    elif t == 3600: ### clark rd
        network.links[3425].close_link_to_newcomers(g=network.g)
        network.links[3425].status = 'closed'
        network.links[20912].close_link_to_newcomers(g=network.g)
        network.links[20912].status = 'closed'
    elif t == 10800: ### neal rd
        for link_id in [22423, 30175, 7854, 30173, 3399, 13378, 16494, 27223]: # old: 9012, 14363
            network.links[link_id].close_link_to_newcomers(g=network.g)
            network.links[link_id].status = 'closed'
    else:
        pass
    
    # run link model; close and open links due to fire
    for link_id, link in network.links.items(): 
        ### link status: open (initial); closed (forever closed); burning (on fire but not closed); burning_closed (on fire and closed); burnt_over (fire moved over and will not be on fire again         )
        link.run_link_model(t, agent_id_dict=network.agents)
        
        ### reopen some links
        if (link.status in ['burning', 'burning_closed']) and (t-link.burnt_time>=3600):
            if link.status == 'burning_closed':
                link.open_link_to_newcomers(g=network.g)
                # print{'{} is set to open'.format(link.link_id)}
            link.status='burnt_over'

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

def one_step(t, data, config):
    # print(t)

    network, links_raster, evacuation_zone, evacuation_buffer, fire_df, fire_array = data['network'], data['links_raster'], data['evacuation_zone'], data['evacuation_buffer'], data['fire_df'], data['fire_array']
    
    scen_nm, simulation_outputs, rout_id, check_traffic_flow_links, reroute_freq, clos_id, special_nodes = config['scen_nm'], config['simulation_outputs'], config['rout_id'], config['check_traffic_flow_links'], config['reroute_freq'], config['clos_id'], config['special_nodes']

    # link closure probability when fire arrives
    clos_id_dict = {'1': 0, '2': 0.15, '3': 0.3}
    link_closure_prob = clos_id_dict[clos_id]

    move = 0
    congested = 0
    step_fitness = None
    
    ### update link travel time before rerouting
    # if (t%reroute_freq==0):
    #     for link in network.links.values():
    #         link.update_travel_time_by_queue_length(network.g)
    
    ### reset link congested counter
    for link in network.links.values():
        link.congested = 0
        ### frequently update link travel time
        if t%100 == 0:
            link.update_travel_time_by_queue_length(network.g)

    # if t%1200==0:
    #     sp = network.g.dijkstra(7481, 7479)
    #     sp_dist = sp.distance(7479)
    #     route_distance = sp_dist
    #     sp_route = sp.route(7479)
    #     route = {}
    #     for (start_nid, end_nid) in sp_route:
    #         route[start_nid] = end_nid
    #     print(t, route_distance, route)

    ### agent model
    t_agent_0 = time.time()
    stopped_agents_list = []
    for agent_id, agent in network.agents.items():
        ### first remove arrived vehicles
        if agent.status == 'arrive':
            network.agents_stopped[agent_id] = (agent.status, t)
            stopped_agents_list.append(agent_id)
            continue
        ### find congested links and active stopped vehicles: spent too long in a link
        current_link = network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]]
        if (agent.current_link_enter_time is not None) and (network.nodes[agent.current_link_start_nid].node_type!='v') and (t-agent.current_link_enter_time>max(3600*0.5, current_link.fft*10)):
            congested += 1
            current_link.congested += 1
        ### shelter in place
        if (agent.current_link_enter_time is not None) and (network.nodes[agent.current_link_start_nid].node_type!='v') and (t-agent.current_link_enter_time>max(3600*2.5, current_link.fft*10)):
            agent.route = {}
            agent.status = 'shelter_a1'
        ### agents need rerouting
        elif rout_id == '1':
            # initial route 
            if (t==0) or (t%reroute_freq==agent.agent_id%reroute_freq):
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            # reroute upon closure
            if (agent.next_link is not None) and (network.links[agent.next_link].status in ['closed', 'burning_closed']) and (t%10 == agent.agent_id%10):
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
        elif rout_id == '2':
            # initial route and fixed interval rerouting
            if (t==0) or (t%reroute_freq==agent.agent_id%reroute_freq and t < 9000) or (t%(6*reroute_freq)==agent.agent_id%(6*reroute_freq) and t > 9000 and np.random.uniform(0,1)>0.5):
                # after cell tower went down, the rerouting frequency becomes half an hour
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            # rerouting at Neal/Skyway when cell tower is out
            if (t>9000 and t <3600*3 and agent.current_link_end_nid ==9334):
                neal_links = [22423, 30174, 27394]
                skyway_links = [22424]
                neal_vehicles = sum([len(network.links[link_id].run_vehicles) + len(network.links[link_id].queue_vehicles) for link_id in neal_links])
                skyway_vehicles = sum([len(network.links[link_id].run_vehicles) + len(network.links[link_id].queue_vehicles) for link_id in skyway_links])
                if neal_vehicles > skyway_vehicles:
                    link = network.links[22423]
                    link.close_link_to_newcomers(g=network.g)
                    previous_distance = agent.route_distance
                    routing_status = agent.get_path(t, g=network.g)
                    link.open_link_to_newcomers(g=network.g)
                    agent.find_next_link(node2link_dict=network.node2link_dict)
                    if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            # rerouting at Neal/Skyway shortly after Neal road is closed
            elif (t>=10800 and t<10800+600 and agent.current_link_end_nid ==9334):
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            # the information of Neal closure is broadcaseted to the rest of the network in 10 minutes
            elif (t==10800+600) and (3790 in agent.route.keys()):
                ### cell tower lost. Neal road closed information took 10 min to arrive at those planning to use it.
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            # reroute upon closure
            elif (t<9000) and (agent.next_link is not None) and (network.links[agent.next_link].status in ['closed', 'burning_closed']) and (t%10 == agent.agent_id%10):
                # print(agent.agent_id, agent.route_distance, agent.status)
                previous_distance = agent.route_distance
                routing_status = agent.get_path(t, g=network.g)
                agent.find_next_link(node2link_dict=network.node2link_dict)
                if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
            elif (t>=9000) and (agent.next_link is not None) and (network.links[agent.next_link].status in ['closed', 'burning_closed']) and (t%10 == agent.agent_id%10):
                previous_distance = agent.route_distance
                if network.links[agent.next_link].status == 'burning_closed':
                    detour_distance, look_ahead_distance, look_ahead_reroute_distance = agent.get_partial_path(t, g=network.g, link_id_dict=network.links, node2link_dict=network.node2link_dict)
                    if (agent.status == 'enroute') and ((agent.route_distance > previous_distance*2) or (look_ahead_reroute_distance>look_ahead_distance*4)): agent.status = 'shelter_a2'
                else:
                    routing_status = agent.get_path(t, g=network.g)
                    if (agent.status == 'enroute') and (agent.route_distance > previous_distance*2): agent.status = 'shelter_a2'
                agent.find_next_link(node2link_dict=network.node2link_dict)
                # if detour_distance > look_ahead_distance*5: agent.status = 'shelter_a2'
            else:
                pass
        else:
            print('invalid rerouting scenarios')
        ### load vehicles
        agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
        ### remove passively sheltered vehicles immediately, no need to wait for node model
        if agent.status in ['shelter_p', 'shelter_a1', 'shelter_a2']:
            current_link.queue_vehicles = [v for v in current_link.queue_vehicles if v!=agent_id]
            current_link.run_vehicles = [v for v in current_link.run_vehicles if v!=agent_id]
            network.agents_stopped[agent_id] = (agent.status, t)
            stopped_agents_list.append(agent_id)
            print(t, agent_id, agent.status, current_link.link_id)
            # if current_link.link_id == 'n4299_vl': print(current_link.queue_vehicles)
            # print(agent.agent_id, current_link.link_id, current_link.queue_vehicles)
        # if (t%120==0) and (agent.agent_id==3595): print(agent.agent_id, agent.route)
    for agent_id in stopped_agents_list:
        del network.agents[agent_id]
    t_agent_1 = time.time()
    # if t>100:
    #     sys.exit(0)

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network, links_raster, reroute_freq, link_closure_prob=link_closure_prob, fire_array=fire_array)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    check_traffic_flow_links_dict = {link_pair: 0 for link_pair in check_traffic_flow_links}
    network, move, check_traffic_flow_links_dict = node_model(t, network, move, check_traffic_flow_links_dict, special_nodes=special_nodes)
    t_node_1 = time.time()

    ### metrics
    if t%120 == 0:
        arrival_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='arrive'])
        active1_shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_a1'])
        active2_shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_a2'])
        passive_shelter_cnts = len([agent_id for agent_id, agent_info in network.agents_stopped.items() if agent_info[0]=='shelter_p'])
        if len(network.agents)==0:
            logging.info("all agents arrive at destinations")
            return step_fitness, network
        # vehicle locations
        veh_loc = [network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]].midpoint for agent in network.agents.values()]
        # fire_loc = [link.midpoint for link in network.links.values() if link.status in ['burning_closed', 'burnning']]
        fire_loc = []
        in_fire_cnts = 0
        for link in network.links.values():
            if link.status in ['burning_closed', 'burning']:
                fire_loc.append(link.midpoint)
                in_fire_cnts += len(link.run_vehicles)+len(link.queue_vehicles)
                # print(link.link_id, link.geometry, link.midpoint)
        # temporarily safe
        outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts = outside_polygon(evacuation_zone, evacuation_buffer, veh_loc)
        if len(fire_loc) == 0:
            vehicle_fire_distance = np.ones(len(veh_loc))*100000
        else:
            vehicle_fire_distance = haversine.scipy_point_to_vertex_distance_positive(np.array(veh_loc), np.array(fire_loc))
        # print(t, len(fire_loc), veh_loc[0:10], fire_loc[0:10])
        # sys.exit(0)
        avg_fire_dist = np.mean(vehicle_fire_distance)
        # in_fire_cnts = sum([len(link.run_vehicles)+len(link.queue_vehicles) for link in network.links.values if link.status in ['burning_closed', 'burnning']])
        ### arrival
        with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm),'a') as t_stats_outfile:
            t_stats_outfile.write(",".join([str(x) for x in [t, arrival_cnts, active1_shelter_cnts, active2_shelter_cnts, passive_shelter_cnts, move, congested, round(avg_fire_dist,2), in_fire_cnts, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts]]) + "\n")
        ### transfer
        with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(scen_nm), 'a') as transfer_stats_outfile:
            transfer_stats_outfile.write("{},".format(t) + ",".join([str(check_traffic_flow_links_dict[(il, ol)]) for (il, ol) in check_traffic_flow_links])+"\n")

        ### fitness metric
        c = in_fire_cnts
    
    ### stepped outputs
    if t%1200==0:
        link_output = pd.DataFrame(
            [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles), round(link.travel_time, 2), link.total_vehicles_left, link.congested) for link in network.links.values() if (link.link_type!='v') and (len(link.queue_vehicles)+len(link.run_vehicles)+link.total_vehicles_left+link.congested>0)], columns=['link_id', 'q', 'r', 't', 'tot_left', 'congested'])
        link_output.to_csv(scratch_dir + simulation_outputs + '/link_stats/link_stats_{}_t{}.csv'.format(scen_nm, t), index=False)
    if t%1200==0:    
        ### node agent cnts
        node_agent_cnts = pd.DataFrame(
            [(agent.current_link_end_nid, agent.status, 1) for agent in network.agents.values()], columns=['node_id', 'status', 'cnt']).groupby(['node_id', 'status']).agg({'cnt': np.sum}).reset_index()
        node_agent_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_agent_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)
        ### node move cnts
        node_move_cnts = pd.DataFrame([(node.node_id, node.node_move, node.x, node.y) for node in network.nodes.values() if (node.node_type!='v') and (node.node_move>0)], columns=['node_id', 'node_move', 'x', 'y'])
        node_move_cnts.to_csv(scratch_dir + simulation_outputs + '/node_stats/node_move_cnts_{}_t{}.csv'.format(scen_nm, t), index=False)

    if t%120==0: 
        burning_links = [link_id for link_id, link in network.links.items() if link.status=='burning']
        burning_closed_links = [link_id for link_id, link in network.links.items() if link.status=='burning_closed']
        burnt_over_links = [link_id for link_id, link in network.links.items() if link.status=='burnt_over']
        closed_links = [link_id for link_id, link in network.links.items() if link.status=='closed']
        logging.info(" ".join([str(i) for i in [t, arrival_cnts, active1_shelter_cnts, active2_shelter_cnts, passive_shelter_cnts, move, congested, '|', len(burning_links), len(burning_closed_links), len(burnt_over_links), len(closed_links), '|', round(avg_fire_dist,2), in_fire_cnts, outside_evacuation_zone_cnts, outside_evacuation_buffer_cnts, round(t_agent_1-t_agent_0, 2), round(t_node_1-t_node_0, 2), round(t_link_1-t_link_0, 2)]]) + " " + str(len(veh_loc)))
    return step_fitness, network

def preparation(random_seed=None, vphh_id='123', dept_id='2', clos_id='2', contra_id='0', rout_id='2', scen_nm=None):+
    ### logging and global variables

    network_file_edges = work_dir + '/network_inputs/butte_edges_sim.csv'
    network_file_nodes = work_dir + '/network_inputs/butte_nodes_sim.csv'
    network_file_special_nodes = '/network_inputs/butte_special_nodes.json'
    network_file_edges_raster = '/network_inputs/butte_edges_sim.tif'
    demand_files = [work_dir + "/demand_inputs/od_r{}_vphh{}_dept{}.csv".format(random_seed, vphh_id, dept_id)]
    simulation_outputs = '' ### scratch_folder
    project_location = work_dir
    if contra_id=='0': cf_files = []
    elif contra_id=='3': cf_files = [work_dir + '/network_inputs/contraflow_skyway_3.csv']
    elif contra_id=='4': cf_files = [work_dir + '/network_inputs/contraflow_skyway_4.csv']
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
    network.dataframe_to_network(project_location=project_location, network_file_edges = network_file_edges, network_file_nodes = network_file_nodes, cf_files = cf_files, special_nodes=special_nodes, scen_nm=scen_nm)
    network.add_connectivity()
    ### traffic signal
    # network.links[21671].capacity /= 2

    ### demand
    network.add_demand(demand_files = demand_files)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    ### evacuation zone
    evacuation_zone_gdf = gpd.read_file(work_dir+'/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp').to_crs('epsg:26910')
    evacuation_zone_gdf = evacuation_zone_gdf.loc[evacuation_zone_gdf['id']<=14].copy()
    evacuation_zone = evacuation_zone_gdf['geometry'].unary_union
    evacuation_buffer = evacuation_zone_gdf['geometry'].buffer(1609).unary_union
    logging.info('Evacuation zone is {:.2f} km2, considering 1 mile buffer it is {:.2f} km2'.format(evacuation_zone.area/1e6, evacuation_buffer.area/1e6))

    ### process the fire information
    fire_df = pd.read_csv(work_dir + "/demand_inputs/simulation_fire_locations.csv")
    fire_df = fire_df[fire_df['type'].isin(['pentz', 'clark', 'neal'])]
    ### fire arrival time
    ### additional fire spread information will be considered later
    fire_array = rio.open(work_dir + '/demand_inputs/fire_cawfe/cawfe_small_spot_fire_r{}_445.tif'.format(random_seed)).read(1)
    
    ### time step output
    with open(scratch_dir + simulation_outputs + '/t_stats/t_stats_{}.csv'.format(scen_nm), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['t', 'arr', 'active1_shelter', 'active2_shelter', 'passive_shelter', 'move', 'congested', 'avg_fdist', 'in_fire_cnts', 'out_evac_zone_cnts', 'out_evac_buffer_cnts'])+"\n")
    ### track the traffic flow from the following link pairs
    check_traffic_flow_links = [(29,33)]
    with open(scratch_dir + simulation_outputs + '/transfer_stats/transfer_stats_{}.csv'.format(
        scen_nm), 'w') as transfer_stats_outfile:
        transfer_stats_outfile.write("t,"+",".join(['{}-{}'.format(il, ol) for (il, ol) in check_traffic_flow_links])+"\n")

    return {'network': network, 'links_raster': links_raster, 'evacuation_zone': evacuation_zone, 'evacuation_buffer': evacuation_buffer, 'fire_df': fire_df, 'fire_array': fire_array}, {'check_traffic_flow_links': check_traffic_flow_links, 'scen_nm': scen_nm, 'simulation_outputs': simulation_outputs, 'rout_id': rout_id, 'clos_id': clos_id, 'reroute_freq': reroute_freq, 'special_nodes': special_nodes}

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
