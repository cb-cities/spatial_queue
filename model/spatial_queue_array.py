#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point, LineString

### usr
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, abs_path)
sys.path.insert(0, "/home/bingyu/Documents")
from sp import interface

np.random.seed(1)

class Network:
    def __init__(self):
        self.nodes = None
        self.links = None
        ### no conflict movement directions by node
        self.no_conflict_moves = dict()
        ### simulation/shortest path network
        self.g = None
        ### some frequently used columns
        self.link_nid_dict = dict()
        self.link_length_dict = dict()
        self.link_fft_dict = dict()
        self.link_lanes_dict = dict()
        self.link_capacity_dict = dict()
    
    def prepare_nodes(self, nodes_df, ods_df):
        ### nodes_df must contain the following columns
        self.nodes = nodes_df[['node_id', 'lon', 'lat']].copy()
        ### if virtual node is hard coded in the input data
        self.nodes['node_type'] = 'real'

        ### add virtual nodes
        virtual_nodes_df = self.nodes.loc[self.nodes['node_id'].isin(np.unique(ods_df['origin_nid']))].copy()
        virtual_nodes_df['node_id'] = virtual_nodes_df['node_id'].apply(lambda x: 'vn_{}'.format(x))
        virtual_nodes_df['node_type'] = 'virtual'
        self.nodes = pd.concat([self.nodes, virtual_nodes_df])
        return virtual_nodes_df
    
    def prepare_links(self, links_df, virtual_nodes_df):
        ### links must contain the following columns; geometry must be in epsg:4326
        self.links = links_df[['link_id', 'start_node_id', 'end_node_id', 'geometry']].copy()
        self.links['link_type'] = 'real'
        
        ### add attributes
        self.links = gpd.GeoDataFrame(self.links, crs='epsg:4326', geometry=links_df['geometry'].map(loads)).to_crs('epsg:3857')
        try: self.links['length'] = links_df['length'] 
        except KeyError:
            self.links['length'] = self.links['geometry'].length
            print('link length not specified; derive from geometry instead.')
        try: self.links['maxmph'] = links_df['maxmph']
        except KeyError:
            self.links['maxmph'] = 25
            print('link speed limit (mph); use 25mph.')
        try: self.links['lanes'] = links_df['lanes']
        except KeyError:
            self.links['lanes'] = 1
            print('link length not specified; assume one lane.')
        
        ### add virtual links
        virtual_links_df = virtual_nodes_df[['node_id', 'lon', 'lat']].copy()
        virtual_links_df['start_node_id'] = virtual_links_df['node_id']
        virtual_links_df['end_node_id'] = virtual_links_df['node_id'].apply(lambda x: int(x.replace('vn_', ''))) ### remove the 'vn_' from node id
        virtual_links_df['link_id'] = virtual_links_df['node_id'].apply(lambda x: x.replace('vn_', 'vl_'))
        virtual_links_df['link_type'] = 'virtual'
        virtual_links_df['geometry'] = None
        virtual_links_df['length'] = 1
        virtual_links_df['maxmph'] = 1000
        virtual_links_df['lanes'] = 1e8

        ### add other attributes
        self.links = pd.concat([self.links, virtual_links_df[self.links.columns]])
        self.links['fft'] = self.links['length']/(self.links['maxmph']*1609/3600)
        self.links['capacity'] = self.links['lanes'] * 1000/3600
        self.links['storage'] = self.links['length'] * self.links['lanes']
        self.links['storage'] = np.where(self.links['storage']<18, 18, self.links['storage'])
        self.links = self.links[['link_id', 'start_node_id', 'end_node_id', 'link_type', 'length', 'lanes', 'maxmph', 'fft', 'capacity', 'storage', 'geometry']]
 
    def node_movements(self):

        link_out_coord = dict()
        link_in_coord = dict()
        node_coords = dict()
        incoming_links = {n: [] for n in self.nodes['node_id'].values.tolist()}
        outgoing_links = {n: [] for n in self.nodes['node_id'].values.tolist()}
        
        for link in self.links[self.links['link_type']=='real'].itertuples():
            ### virtual incoming links not considered here
            link_id = getattr(link, 'link_id')
            link_geometry = getattr(link, 'geometry')
            link_snid = getattr(link, 'start_node_id')
            link_enid = getattr(link, 'end_node_id')
            
            ### near end coordinates
            ### first upstream node except from the start node
            link_in_coord[link_id] = link_geometry.coords[1]
            ### last downstream node except from the end node
            link_out_coord[link_id] = link_geometry.coords[-2]
            
            ### node coordinates
            if link_snid not in node_coords: node_coords[link_snid] = link_geometry.coords[0]
            if link_enid not in node_coords: node_coords[link_enid] = link_geometry.coords[-1] 
            
            ### node to all links
            incoming_links[link_enid].append(link_id)
            outgoing_links[link_snid].append(link_id)
        
        #display(self.nodes[self.nodes['node_id']==194563])
        for node in self.nodes[self.nodes['node_type']=='real'].itertuples():
            node_id = getattr(node, 'node_id')
            nc = node_coords[node_id]
            node_movements_info = dict()
            for il in incoming_links[node_id]:
                for ol in outgoing_links[node_id]:
                    ### virtual incoming links not considered here
                    ilc, olc = link_out_coord[il], link_in_coord[ol]
                    move_1 = np.arctan2(nc[1]-ilc[1], nc[0]-ilc[0])*180/np.pi
                    move_2 = np.arctan2(olc[1]-nc[1], olc[0]-nc[0])*180/np.pi
                    if move_1 < move_2: move_2 -= 360
                    if (move_1 - move_2)<=45 or (move_1 - move_2)>315: move_dir='sa'
                    elif (move_1 - move_2)>45 and (move_1 - move_2)<170: move_dir='r'
                    elif (move_1 - move_2)>=170 and (move_1 - move_2)<=315: move_dir='left' ### left or u turn
                    else: move_dir = None
                    #if node_id==29: print(il, ol, ilc, nc, olc, move_1, move_2, move_dir)
                    node_movements_info[(il, ol)] = [ilc, olc, move_dir]
            ### pair-wise evaluation of movements in this intersection
            for (il, ol), [ilc, olc, move_dir] in node_movements_info.items():
                for (il_, ol_), [ilc_, olc_, move_dir_] in node_movements_info.items():
                    if (il==il_) and (ol==ol_):
                        ### same direction can go together
                        self.no_conflict_moves[(il, ol, il_, ol_)] = 1
                        continue
                    if (il==il_) and (ol!=ol_):
                        ### same in direction but different out direction
                        if (move_dir in ['left']) or (move_dir_ in ['left']):
                            continue
                        else: self.no_conflict_moves[(il, ol, il_, ol_)] = 1
                    ### find opposite direction
                    move_1 = np.arctan2(nc[1]-ilc[1], nc[0]-ilc[0])*180/np.pi
                    move_2 = np.arctan2(nc[1]-ilc_[1], nc[0]-ilc_[0])*180/np.pi
                    if move_1 < move_2: move_2 -= 360
                    if (move_1-move_2)>=170 and (move_1-move_2)<=190:
                        ### opposite direction
                        if (move_dir in ['left']) and (move_dir_ in ['sa', 'r']):
                            continue
                        if (move_dir in ['sa', 'r']) and (move_dir_ in ['left']):
                            continue
                        #if (move_dir in ['left']) and (move_dir_ in ['left']):
                        #    continue
                        self.no_conflict_moves[(il, ol, il_, ol_)] = 1
                    else:
                        continue
    
    def prepare_network(self, nodes_df, links_df, ods_df):
        
        ### add virtual nodes and virtual links
        ### only return virtual nodes as it will be used in the creation of virtual links in the next function
        virtual_nodes_df = self.prepare_nodes(nodes_df, ods_df) 
        self.prepare_links(links_df, virtual_nodes_df)
        
        ### find no conflict movements
        self.node_movements()

        ### some lookup or calculation variables
        self.g = interface.from_dataframe(self.links[self.links['link_type']=='real'], 'start_node_id', 'end_node_id', 'fft')
        self.link_nid_dict = {(getattr(link, 'start_node_id'), getattr(link, 'end_node_id')): getattr(link, 'link_id') for link in self.links.itertuples()}
        self.link_length_dict = {getattr(link, 'link_id'): getattr(link, 'length') for link in self.links.itertuples()}
        self.link_lanes_dict = {getattr(link, 'link_id'): getattr(link, 'lanes') for link in self.links.itertuples()}
        self.link_fft_dict = {getattr(link, 'link_id'): getattr(link, 'fft') for link in self.links.itertuples()}
        self.link_capacity_dict = {getattr(link, 'link_id'): getattr(link, 'capacity') for link in self.links.itertuples()}

    def find_next_link(self, agent, agent_routes, just_loaded=False):
        agent_id = getattr(agent, 'agent_id')
        agent_cl_enid = getattr(agent, 'cl_enid')
        ### just loaded
        if just_loaded:
            just_loaded_nl_enid = agent_routes[agent_id][agent_cl_enid]
            just_loaded_nl = self.link_nid_dict[(agent_cl_enid, just_loaded_nl_enid)]
            just_loaded_nl_lanes = self.link_lanes_dict[just_loaded_nl]
            just_loaded_nl_capacity = self.link_capacity_dict[just_loaded_nl]
            return ['vl_{}'.format(agent_cl_enid), 'vn_{}'.format(agent_cl_enid), agent_cl_enid, 0.1, 1e8, 1e8/3.6, just_loaded_nl, just_loaded_nl_enid, just_loaded_nl_lanes, just_loaded_nl_capacity]
        ### current link end is destination
        if agent_cl_enid == getattr(agent, 'destin_nid'):
            return ['vl_sink', agent_cl_enid, 'sink_node', 0.1, 1e8, 1e8/3.6, None, None, None, None]
        agent_nl_enid = getattr(agent, 'nl_enid') #agent_routes[agent_id][agent_cl_enid]
        agent_nl = getattr(agent, 'next_link') #self.link_nid_dict[(agent_cl_enid, agent_nl_enid)]
        agent_nl_fft = self.link_fft_dict[agent_nl]
        agent_nl_lanes = self.link_lanes_dict[agent_nl]
        agent_nl_capacity = self.link_capacity_dict[agent_nl]
        ### next link end is destination, next next link does not exist
        if agent_nl_enid == getattr(agent, 'destin_nid'):
            #return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, agent_nl_lanes, agent_nl_capacity, 'vl_sink', 'sink_node', 1e8, 1e8/3.6]
            return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, 1000, 1e7/3.6, 'vl_sink', 'sink_node', 1e8, 1e8/3.6]
        agent_nnl_enid = agent_routes[agent_id][agent_nl_enid]
        agent_nnl = self.link_nid_dict[(agent_nl_enid, agent_nnl_enid)]
        agent_nnl_lanes = self.link_lanes_dict[agent_nnl]
        agent_nnl_capacity = self.link_capacity_dict[agent_nnl]
        return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, agent_nl_lanes, agent_nl_capacity, agent_nnl, agent_nnl_enid, agent_nnl_lanes, agent_nnl_capacity]

    def run_link_model(self, agents=None, t=None):

        agents_df = agents.agents
        
        ### running vehicles joins queue
        agents_df['agent_status'] = np.where(
            (agents_df['agent_status']==1) &
            (agents_df['cl_fft']<(t-agents_df['current_link_enter_time'])),
            2, agents_df['agent_status'])

        ### summarize from agents_df
        link_undeparted_dict = agents_df.loc[agents_df['agent_status']==0].groupby('current_link').size().to_dict()
        link_queue_dict = agents_df.loc[agents_df['agent_status']==2].groupby('current_link').size().to_dict()
        link_run_dict = agents_df.loc[agents_df['agent_status']==1].groupby('current_link').size().to_dict()
 
        ### update links
        self.links['undeparted'] = self.links['link_id'].map(link_undeparted_dict).fillna(0)
        self.links['queue'] = self.links['link_id'].map(link_queue_dict).fillna(0)
        self.links['run'] = self.links['link_id'].map(link_run_dict).fillna(0)
        self.links['storage_remain'] = self.links['storage'] - (self.links['run'] + self.links['queue'])*8

        return agents_df
    
    def run_node_model(self, agents=None, t=None, special_nodes=None):
        
        agents_df = agents.agents
        agent_routes = agents.agent_routes
        
        ### queue vehicles
        go_agents_df = agents_df.loc[
            agents_df['agent_status']==2].copy()
        if go_agents_df.shape[0] == 0:
            ### no queuing agents 
            return agents_df
        
        ### choose primary movement direction
        ### 1. first come first serve
        #go_agents_df = go_agents_df.sort_values(by='current_link_enter_time', ascending=True)
        ### 2. random (e.g., a random traffic light)
        go_agents_df = go_agents_df.sample(frac=1)
        ### 3.highest control capacity can go
        #go_agents_df = go_agents_df.sort_values(by='control_capacity', ascending=True)
        ### user-specified movement priority at nodes (e.g., roundabouts)
        #go_agents_midx = pd.MultiIndex.from_frame(go_agents_df[['current_link', 'next_link']])
        #go_agents_df['default_priority'] = 3
        #go_agents_df['default_priority'] = np.where(go_agents_midx.isin(
        #    special_nodes['first_priority']), 1, go_agents_df['default_priority'])
        #go_agents_df['default_priority'] = np.where(go_agents_midx.isin(
        #    special_nodes['second_priority']), 2, go_agents_df['default_priority'])
        #go_agents_df = go_agents_df.sort_values(by='default_priority', ascending=True)
        ### the results: directions of the first vehicle in the queue after sorting
        go_agents_df['primary_cl'] = go_agents_df.groupby('cl_enid', sort=False)['current_link'].transform('first')
        go_agents_df['primary_nl'] = go_agents_df.groupby('cl_enid', sort=False)['next_link'].transform('first')
        
        ### limit by capacity
        go_agents_df['control_capacity'] = np.floor(
            go_agents_df[['cl_capacity', 'nl_capacity']].min(axis=1) ### mostly zero (except highways)
            + np.random.random(size=go_agents_df.shape[0]))
        #print(go_agents_df.head(1))
        go_agents_df['control_capacity'] = go_agents_df.groupby(
            ['current_link', 'next_link'])['control_capacity'].transform('first')
        ### filter for vehicles that can move according to the control capacity
        #print(go_agents_df.head(1))
        go_agents_df['current_link_queue_time'] = t - go_agents_df['cl_fft'] - go_agents_df['current_link_enter_time']
        #print(go_agents_df.head(1))
        go_agents_df = go_agents_df.sort_values(by='current_link_queue_time', ascending=False)
        #print(go_agents_df.head(1))
        go_agents_df['clnl_position'] = go_agents_df.groupby(['current_link', 'next_link'], sort=False).cumcount()
        go_agents_df = go_agents_df.loc[
            go_agents_df['clnl_position']<go_agents_df['control_capacity']]
        if go_agents_df.shape[0] == 0:
            return agents_df
        
        ### all nodes where the primary movement can go
        selected_nodes = go_agents_df.loc[
            (go_agents_df['current_link']==go_agents_df['primary_cl']) &
            (go_agents_df['next_link']==go_agents_df['primary_nl']), 'cl_enid'
        ].values
        go_agents_df = go_agents_df.loc[go_agents_df['cl_enid'].isin(selected_nodes)].copy()
        if go_agents_df.shape[0] == 0:
            return agents_df
        
        ### label non-conflict movements
        go_agents_df['no_conflict'] = list(map(self.no_conflict_moves.get, go_agents_df[['primary_cl', 'primary_nl', 'current_link', 'next_link']].to_records(index=False).tolist()))
        ### try if it is faster?
        #go_agents_midx = pd.MultiIndex.from_frame(go_agents_df[['primary_cl', 'primary_nl', 'current_link', 'next_link']])
        #go_agents_df['no_conflict'] = np.where(go_agents_midx.isin(self.no_conflict_moves.keys()), 1, 0)
        ### Coming from virtual source or going to virtual sink. Assume no conflict with any direction.
        go_agents_df['no_conflict'] = np.where(
            (go_agents_df['cl_lanes']>=100) | 
            (go_agents_df['nl_lanes']>=100), 1, go_agents_df['no_conflict'])
        go_agents_df = go_agents_df.loc[go_agents_df['no_conflict']==1].copy()
        if go_agents_df.shape[0] == 0:
            return agents_df

        ### find next link storage constraints, changes each time step
        links_to_use_df = self.links.loc[(self.links['link_type']=='real') & (self.links['link_id'].isin(go_agents_df['current_link']) | self.links['link_id'].isin(go_agents_df['next_link']))]
        links_storage_remain_dict = dict(zip(links_to_use_df['link_id'], links_to_use_df['storage_remain']))
        ### no constraints if not in dictionary; meaning the links are either virtual source or sink
        go_agents_df['nl_storage_remain'] = go_agents_df['next_link'].map(links_storage_remain_dict).fillna(1e5)
        
        ### keep only vehicles that the link outflow capacity allows (next link inflow is not controlling as two movements goign to the same downstream link is usually considered conflicting)
        ### no sorting again, keep the same with the order random or higher control capacity first, etc.
        ### MAYBE PROBLEMATIC FOR HIGHWAYS
        go_agents_df['cl_position'] = go_agents_df.groupby('current_link', sort=False).cumcount()
        go_agents_df = go_agents_df.loc[
            (go_agents_df['cl_position']<go_agents_df['control_capacity']) |
            (go_agents_df['nl_lanes']>=100)
        ].copy()
        go_agents_df['nl_position'] = go_agents_df.groupby('next_link', sort=False).cumcount()
        go_agents_df = go_agents_df.loc[
            (go_agents_df['nl_position']<go_agents_df['control_capacity']) &
            (go_agents_df['nl_position']*8<go_agents_df['nl_storage_remain'])
        ].copy()
        
        if go_agents_df.shape[0] == 0:
            return agents_df
        #test = go_agents_df.loc[go_agents_df['cl_enid']==17]
        #if test.shape[0]>0:
        #    print(t, test[['current_link', 'next_link', 'control_capacity', 'cl_capacity', 'nl_capacity']].values.tolist())
        
        ### update agent position
        go_ids = go_agents_df['agent_id']
        agent_clnl_update = {getattr(agent, 'agent_id'): self.find_next_link(agent, agent_routes) for agent in go_agents_df.itertuples()}
        #if len(keep)>0:
        #    print(agent_clnl_update[keep[0]])
        agents_df.loc[agents_df['agent_id'].isin(go_ids), ['current_link', 'cl_snid', 'cl_enid', 'cl_fft', 'cl_lanes', 'cl_capacity', 'next_link', 'nl_enid', 'nl_lanes', 'nl_capacity']] = pd.DataFrame(agents_df.loc[agents_df['agent_id'].isin(go_ids), 'agent_id'].map(agent_clnl_update).to_list()).values
        ### set agent status to run
        agents_df.loc[agents_df['agent_id'].isin(go_ids), 'agent_status'] = 1
        agents_df.loc[agents_df['agent_id'].isin(go_ids), 'current_link_enter_time'] = t
        agents_df['agent_status'] = np.where(agents_df['current_link']=='vl_sink', -1, agents_df['agent_status'])

        return agents_df
        

class Agents:
    def __init__(self):
        ### hold all agents in a class
        self.agents = None ### a pandas dataframe
        self.agent_routes = dict() ### key: agent_id, value: route

    def prepare_agents(self, ods_df):
        ### ods_df should have at least the following columns
        ### `nid` correspond to node_id
        self.agents = ods_df[['origin_nid', 'destin_nid']].copy()
        ### other columns to initialize
        self.agents['agent_status'] = 0 ### 0: unloaded; 1: run; 2:queue, -1: arrive; -2:no_route
        try: self.agents['agent_id'] = ods_df['agent_id']
        except KeyError:
            self.agents['agent_id'] = np.arange(self.agents.shape[0])
            print('Agent IDs not specified; use sequential number.')
        try: self.agents['departure_time'] = ods_df['departure_time']
        except KeyError:
            self.agents['departure_time'] = 0
            print('agents departure times not specified; assume leave immediately.')
        for agent_attribute in ['current_link', 'current_link_enter_time', 'cl_snid', 'cl_enid', 'cl_fft', 'cl_lanes', 'next_link', 'nl_enid', 'nl_lanes']:
            self.agents[agent_attribute] = np.nan

    def get_agent_routes(self, origin, destin, g):
        ### note this applies to individual agents
        sp = g.dijkstra(origin, destin)
        sp_dist = sp.distance(destin)
        if sp_dist > 1e8:
            sp.clear()
            route = {}
        else:
            route = {start_node_id: end_node_id for (start_node_id, end_node_id) in sp.route(destin)}
            #if 16286 in route.keys():
            #    print(route)
            sp.clear()
        return route

    def update_route_info(self, reroute_agents, network, just_loaded=False):

        unarrived_list = []
        agent_clnl_update = dict()
        for agent in reroute_agents.itertuples():
            ### agent id
            agent_id = getattr(agent, 'agent_id')
            ### get initial or new routes
            route = self.get_agent_routes(int(getattr(agent, 'cl_enid')), int(getattr(agent, 'destin_nid')), network.g)
            ### update route
            self.agent_routes[agent_id] = route
            ### no path
            if len(route)==0: 
                unarrived_list.append(agent_id)
                print('no route for {}'.format(agent_id))
            else: agent_clnl_update[agent_id] = network.find_next_link(agent, self.agent_routes, just_loaded=True)
        if len(unarrived_list)>0:
            self.agents.loc[self.agents['agent_id'].isin(unarrived_list), 'agent_status'] = -2
        self.agents.loc[self.agents['agent_id'].isin(reroute_agents['agent_id']), 
                        ['current_link', 'cl_snid', 'cl_enid', 'cl_fft', 'cl_lanes', 'cl_capacity',
                         'next_link', 'nl_enid', 'nl_lanes', 'nl_capacity'
                        ]] = pd.DataFrame(reroute_agents['agent_id'].map(agent_clnl_update).to_list()).values

    def load_agents(self, network=None, t=None):
        ### load agent
        load_ids = t == self.agents['departure_time']
        if np.sum(load_ids) == 0:
            return
        self.agents['cl_enid'] = np.where(load_ids, self.agents['origin_nid'], self.agents['cl_enid'])
        ### loaded to the queue on virtual source links
        self.agents['agent_status'] = np.where(load_ids, 2, self.agents['agent_status'])
        self.agents['current_link_enter_time'] = np.where(load_ids, t, self.agents['current_link_enter_time'])
        #print(self.agents[load_ids])
        self.update_route_info(self.agents[load_ids], network, just_loaded=True)

    def reroutes(self, network=None, t=None, reroute_frequency=None):
        
        if reroute_frequency is None:
            ### no reroute
            return
        else:
            ### staggered rerouting
            reroute_agents = self.agents[
                (self.agents['agent_status'].isin([1, 2])) &
                (self.agents['agent_id']%reroute_frequency==t)]
        ### update routes
        self.update_route_info(reroute_agents, network)


class Simulation():
    def __init__(self):
        self.agents = None
        self.network = None


    def initialize_simulation(self, nodes_df, links_df, ods_df):
        
        ### initialize network
        self.network = Network()
        self.network.prepare_network(nodes_df, links_df, ods_df)
        
        ### initialize demand
        self.agents = Agents()
        self.agents.prepare_agents(ods_df)
    

    def run_one_step(self, t, reroute_frequency=None, special_nodes=None):
        
        ### load agents that are ready to departure; find routes and next links for them
        self.agents.load_agents(network=self.network, t=t)
        
        ### reroute agents
        self.agents.reroutes(network=self.network, t=t, reroute_frequency=reroute_frequency)
        self.agents.agents = self.network.run_link_model(agents=self.agents, t=t)
        self.agents.agents = self.network.run_node_model(agents=self.agents, t=t, special_nodes=special_nodes)




