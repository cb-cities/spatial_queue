import os
import sys
import time 
import random 
import numpy as np
import pandas as pd 
from ctypes import c_double
from multiprocessing import Pool

from squeue import network as sq_network
from squeue import agent as sq_agent

absolute_path = '/home/bingyu/Documents/spatial_queue'
sys.path.insert(0, absolute_path+'/../')
from sp import interface 

def read_network(network_file_edges=None, network_file_nodes=None, simulation_outputs=None, scen_nm=''):

    links_df0 = pd.read_csv(absolute_path+network_file_edges)
    ### tertiary and below
    links_df0['lanes'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified']), 25, links_df0['maxmph'])
    ### primary
    links_df0['lanes'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 1, links_df0['lanes'])
    links_df0['maxmph'] = np.where(links_df0['type'].isin(['primary', 'primary_link']), 55, links_df0['maxmph'])

    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237
    links_df0['capacity'] = 2000*links_df0['lanes']
    links_df0 = links_df0[['edge_id_igraph', 'start_igraph', 'end_igraph', 'lanes', 'capacity', 'maxmph', 'fft', 'length', 'geometry']]

    nodes_df0 = pd.read_csv(absolute_path+network_file_nodes)
    network = sq_network.Network(nodes_df0, links_df0)

    return network

def demand(demand_files=None, phased_flag=False, network=None):

    all_od_list = []
    for demand_file in demand_files:
        od = pd.read_csv(absolute_path + demand_file)
        
        if 'agent_id' not in od.columns: od['agent_id'] = np.arange(od.shape[0])   
        if phased_flag == False: od['dept_time'] = 0
        else: od['dept_time'] = np.random.randint(low=0, high=3600*5, size=od.shape[0])
        
        od['origin_nid'] = od['origin_osmid'].apply(lambda x: network.nodes_osmid_dict[x])
        od['destin_nid'] = od['destin_osmid'].apply(lambda x: network.nodes_osmid_dict[x])
        all_od_list.append(od)

    all_od = pd.concat(all_od_list, sort=False, ignore_index=True)
    all_od = all_od.sample(frac=1).reset_index(drop=True) ### randomly shuffle rows
    # print('total numbers of agents from file ', all_od.shape)
    all_od = all_od.iloc[0:5000].copy()
    # print('total numbers of agents taken ', all_od.shape)
    
    agents = sq_agent.AllAgents(od_df=all_od, network=network)
    return agents

def map_sp(agent_id):
    
    subp_agent = agent_id_dict[agent_id]
    subp_agent.get_path()
    return (agent_id, subp_agent)

def route(scen_nm=''):
    
    ### Build a pool
    process_count = 10
    pool = Pool(processes=process_count)

    ### Find shortest pathes
    t_odsp_0 = time.time()
    map_agent = [k for k, v in agent_id_dict.items() if v.cle != None]
    res = pool.imap_unordered(map_sp, map_agent)

    ### Close the pool
    pool.close()
    pool.join()
    cannot_arrive = 0
    for (agent_id, subp_agent) in list(res):
        agent_id_dict[agent_id].find_route = subp_agent.find_route
        agent_id_dict[agent_id].route_igraph = subp_agent.route_igraph
        if subp_agent.find_route=='n_a': cannot_arrive += 1
    t_odsp_1 = time.time()

    if cannot_arrive>0: print('{} out of {} cannot arrive'.format(cannot_arrive, len(agent_id_dict)))
    return t_odsp_1-t_odsp_0, len(map_agent)

def main(random_seed=0, transfer_s=None, transfer_e=None, node_demand=None):
    random.seed(random_seed)
    np.random.seed(random_seed)
    global g, agent_id_dict
    
    reroute_flag = True
    reroute_freq = 10 ### sec
    link_time_lookback_freq = 20 ### sec
    phased_flag = False
    scen_nm = 'class_test'
    network_file_edges = '/projects/bolinas_stinson_beach/network_inputs/osm_edges.csv'
    network_file_nodes = '/projects/bolinas_stinson_beach/network_inputs/osm_nodes.csv'
    demand_files = ['/projects/bolinas_stinson_beach/demand_inputs/bolinas_od_{}_per_origin.csv'.format(node_demand)]
    simulation_outputs = '/projects/bolinas_stinson_beach/simulation_outputs'

    network = read_network(
        network_file_edges = network_file_edges, network_file_nodes = network_file_nodes,
        simulation_outputs = simulation_outputs, scen_nm = scen_nm)
    agents = demand(demand_files = demand_files, phased_flag = phased_flag, network=network)
    agent_id_dict = agents.agents
    # g = network.g
    g = interface.readgraph(bytes(absolute_path+simulation_outputs+'/network_sparse.mtx', encoding='utf-8'))

    t_s, t_e = 0, 101
    move = 0
    route_time = 0
    route_cnts = 0
    for t in range(t_s, t_e):
        ### routing
        if (t==0) or (reroute_flag) and (t%reroute_freq == 0):
            ### update link travel time
            network.update_all_links_travel_time(t_now=t, link_time_lookback_freq=link_time_lookback_freq)
            ### route
            rt, rc = route(scen_nm=scen_nm)
            route_time += rt
            route_cnts += rc
        ### load agents
        agents.load_trips(t_now=t)
        ### move one time step, run link model and node model
        n_t_move, n_t_key_loc_flow = network.one_step(t_now=t, agents=agents.agents)
        move += n_t_move
        if t%100==0: print(t, np.sum([1 for a in agents.agents.values() if a.status=='arr']), route_time, route_cnts)
    
    print(np.sum([1 for a in agents.agents.values() if a.status=='arr']))

if __name__ == '__main__':
    # g = None
    # agent_id_dict = None
    main(node_demand=3)