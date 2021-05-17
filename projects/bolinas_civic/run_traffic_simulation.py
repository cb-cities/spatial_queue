import os
import time
import json 
import random
import logging 
import numpy as np
import pandas as pd 
import geopandas as gpd 
from shapely.wkt import loads

from pathlib import Path

### dir
home_dir = os.path.dirname(os.path.abspath(__file__))

### user
from demand_inputs.simple_od import generate_simple_od
from queue_class import Network, Node, Link, Agent

### set random seed
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

def preparation(vphh=None, visitor_cnts=None, scen_nm=None, write_path = None):
    ### logging and global variables

    network_file_edges = '/network_inputs/bolinas_edges_sim.csv'
    network_file_nodes = '/network_inputs/bolinas_nodes_sim.csv'
    network_file_special_nodes = '/network_inputs/bolinas_special_nodes.json'
    demand_files = ['/demand_inputs/od_csv/resident_visitor_od_vphh{}_visitor{}.csv'.format(vphh, visitor_cnts)]
    simulation_outputs = '/simulation_outputs'

    scen_nm = scen_nm
    logging.basicConfig(filename=write_path+simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(scen_nm)
    print('log file created for {}'.format(scen_nm))

    ### network
    with open(home_dir + network_file_special_nodes) as special_nodes_file:
        special_nodes = json.load(special_nodes_file)
    network = Network()
    network.dataframe_to_network(project_location='', network_file_edges = network_file_edges, network_file_nodes = network_file_nodes, special_nodes=special_nodes, scen_nm=scen_nm, write_path = write_path)
    network.add_connectivity()

    ### demand
    network.add_demand(demand_files = demand_files, write_path = write_path)
    logging.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    return {'network': network}, {'scen_nm': scen_nm, 'simulation_outputs': simulation_outputs, 'special_nodes': special_nodes}, {}

def link_model(t, network, link_closed_time=None, closed_mode=None):
    
    # run link model
    for link_id, link in network.links.items(): 
        link.run_link_model(t, agent_id_dict=network.agents)

    return network

def node_model(t, network, special_nodes=None):
    # only run node model for those with vehicles waiting
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles)>0])

    # run node model
    for node_id in node_ids_to_run:
        node = network.nodes[node_id] 
        n_t_move, transfer_links= node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links, agent_id_dict=network.agents, node2link_dict=network.node2link_dict, special_nodes=special_nodes)
        
    return network

def one_step(t, data, config, update_data):

    network = data['network']
    
    scen_nm, simulation_outputs, special_nodes = config['scen_nm'], config['simulation_outputs'], config['special_nodes']

    ### update link travel time before rerouting
    reroute_freq = 300
    
    ### reset link congested counter
    for link in network.links.values():
        link.congested = 0
        if (t%100==0):
            link.update_travel_time_by_queue_length(network.g)

    if t==0:
        for agent_id, agent in network.agents.items():
            network.agents[agent_id].departure_time = np.random.randint(1, 100)

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
            current_link.congested += 1
            if (t-agent.current_link_enter_time>3600*3):
                agent.status = 'shelter_a1'
        ### agents need rerouting
        # initial route 
        if (t==0) or (t%reroute_freq==agent_id%reroute_freq):
            routing_status = agent.get_path(t, g=network.g)
            agent.find_next_link(node2link_dict=network.node2link_dict)
            # if agent_id == 0: print(agent.route)
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
    network = link_model(t, network)
    t_link_1 = time.time()
    
    ### node model
    t_node_0 = time.time()
    network = node_model(t, network, special_nodes=special_nodes)
    t_node_1 = time.time()
        
    # stop
    if len(network.agents)==0:
        logging.info("all agents arrive at destinations")
        return network, 'stop'
    else:
        return network, 'continue'

def run_traffic_simulation(vphh=None, visitor_cnts=None, write_path=None):
    
    # write_path is the directory mimicing the directory /simulation_outputs
    if not write_path:
        write_path = home_dir
        demand_write_path = home_dir + '/demand_inputs'
        print("Writing at default project diretory:", write_path)
    else:
        demand_write_path = write_path + '/demand_inputs'
        print("Writing at alternative diretory:", write_path)

    # Check if all directories are ready
    Path(write_path).mkdir(parents=True, exist_ok=True)
    Path(write_path + '/simulation_outputs/link_weights').mkdir(parents=True, exist_ok=True)
    Path(write_path + '/simulation_outputs/log').mkdir(parents=True, exist_ok=True)
    Path(write_path + '/simulation_outputs/network').mkdir(parents=True, exist_ok=True)
    Path(demand_write_path).mkdir(parents=True, exist_ok=True)
    Path(demand_write_path + '/od_csv').mkdir(parents=True, exist_ok=True)


    ### generate demand
    generate_simple_od(vphh=vphh, visitor_cnts=visitor_cnts, demand_write_path = demand_write_path)

    # base network as the base layer of plotting
    roads_df = pd.read_csv(home_dir + '/network_inputs/bolinas_edges_sim.csv')
    roads_gdf = gpd.GeoDataFrame(roads_df, crs='epsg:4326', geometry=roads_df['geometry'].map(loads)).to_crs(26910)

    # set scenario name
    scen_nm = "vphh{}_visitor{}".format(vphh, visitor_cnts)

    data, config, update_data = preparation(vphh=vphh, visitor_cnts=visitor_cnts, scen_nm=scen_nm, write_path = write_path)

    link_speed_dict = dict()
    link_vehicles_dict = dict()
    link_maxspeed_array = np.array([link.length/(link.fft+0.00001) for link_id, link in data['network'].links.items() if link.link_type != 'v'])
    link_length_array = np.array([(link.length+0.00001) for link_id, link in data['network'].links.items() if link.link_type != 'v'])
    link_id_array = np.array([link_id for link_id, link in data['network'].links.items() if link.link_type != 'v'])
    # print(link_id_array[0:10])

    for t in range(0, 5400):
        # run simulation for one step
        network, status = one_step(t, data, config, update_data)
        link_vehicles_array = np.array([(len(link.run_vehicles) + len(link.queue_vehicles)) for link_id, link in data['network'].links.items() if link.link_type != 'v'])
        link_density_array = link_vehicles_array * 8 / link_length_array
        link_speed_array = np.where(link_density_array>=1, 0, link_maxspeed_array*(1-link_density_array))
        link_speed_dict[t] = link_speed_array.round(2).tolist()
        link_vehicles_dict[t] = link_vehicles_array
    # print(link_speed_dict[199][0:10])

    with open(write_path+'/simulation_outputs/link_weights/link_speed_{}.json'.format(scen_nm), 'w') as outfile:
        json.dump(link_speed_dict, outfile, indent=2)
    
    network_links_dict = dict()
    for link_id, link in network.links.items():
        network_links_dict[link_id] = {'start_nid': link.start_nid, 'end_nid': link.end_nid, 'length': link.length}
    with open(write_path+'/simulation_outputs/network/network_links.json', 'w') as outfile:
        json.dump(network_links_dict, outfile, indent=2)

    node2link_dict = dict()
    for (start_nid, end_nid), link_id in network.node2link_dict.items():
        try:
            node2link_dict[start_nid][end_nid] = link_id
        except KeyError:
            node2link_dict[start_nid] = {end_nid: link_id}
    with open(write_path+'/simulation_outputs/network/node2link_dict.json', 'w') as outfile:
        json.dump(node2link_dict, outfile, indent=2)

