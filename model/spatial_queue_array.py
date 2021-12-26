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

np.random.seed(0)
global movement_order

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
    
    def prepare_nodes(self, nodes_df, ods_df):
        ### nodes_df must contain the following columns
        self.nodes = nodes_df[['node_id', 'lon', 'lat']].copy()
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
        self.links['capacity'] = self.links['lanes'] * 2000/3600
        self.links['storage'] = self.links['length'] * self.links['lanes']
        self.links['storage'] = np.where(self.links['storage']<16, 16, self.links['storage'])
        self.links = self.links[['link_id', 'start_node_id', 'end_node_id', 'link_type', 'length', 'lanes', 'maxmph', 'fft', 'capacity', 'storage', 'geometry']]
 
    def node_movements(self):

        link_out_coord = dict()
        link_in_coord = dict()
        for link in self.links[self.links['link_type']=='real'].itertuples():
            link_id = getattr(link, 'link_id')
            link_geometry = getattr(link, 'geometry')
            link_ss = link.geometry.interpolate(0.01, normalized=True)
            link_ee = link.geometry.interpolate(0.99, normalized=True)
            ### link start coordinates
            (link_sx, link_sy) = link_geometry.coords[0]
            ### link end coordinates
            (link_ex, link_ey) = link_geometry.coords[-1]
            ### near start coordinates
            (link_ssx, link_ssy) = link_ss.coords[0]
            ### near end coordinates
            (link_eex, link_eey) = link_ee.coords[0]
            ### off-set from center line
            link_in_coord[link_id] = [link_ssx + (-link_sy+link_ssy), link_ssy - (-link_sx+link_ssx)]
            link_out_coord[link_id] = [link_eex + (link_ey-link_eey), link_eey - (link_ex-link_eex)]

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def intersect(A, B, C, D):
            ### returns true if line segments AB and CD intersect
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
            
        incoming_links = {n: [] for n in self.nodes['node_id'].values.tolist()}
        outgoing_links = {n: [] for n in self.nodes['node_id'].values.tolist()}

        for link in self.links[self.links['link_type']=='real'].itertuples():
            link_id = getattr(link, 'link_id')
            incoming_links[getattr(link, 'end_node_id')].append(link_id)
            outgoing_links[getattr(link, 'start_node_id')].append(link_id)
            
        for node in self.nodes.itertuples():
            node_id = getattr(node, 'node_id')
            node_movements_info = dict()
            for il in incoming_links[node_id]:
                for ol in outgoing_links[node_id]:
                    ### virtual incoming links not considered here
                    ilc, olc = link_out_coord[il], link_in_coord[ol]
                    node_movements_info[(il, ol)] = ilc, olc
            ### pair-wise evaluation of movements in this intersection
            for (il, ol), [ilc, olc] in node_movements_info.items():
                for (il_, ol_), [ilc_, olc_] in node_movements_info.items():
                    if (il==il_) and (ol==ol_):
                        ### same direction can go together
                        self.no_conflict_moves[(il, ol, il_, ol_)] = 1
                    elif intersect(ilc, olc, ilc_, olc_):
                        ### paths intersect cannot go together
                        continue
                    else:
                        ### no intersection in path can go together
                        self.no_conflict_moves[(il, ol, il_, ol_)] = 1
    
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

    def find_next_link(self, agent, agent_routes, just_loaded=False):
        agent_id = getattr(agent, 'agent_id')
        agent_cl_enid = getattr(agent, 'cl_enid')
        ### just loaded
        if just_loaded:
            just_loaded_nl_enid = agent_routes[agent_id][agent_cl_enid]
            just_loaded_nl = self.link_nid_dict[(agent_cl_enid, just_loaded_nl_enid)]
            just_loaded_nl_fft = self.link_fft_dict[just_loaded_nl]
            return ['vl_{}'.format(agent_cl_enid), 'vn_{}'.format(agent_cl_enid), agent_cl_enid, 0.1, 1e8, just_loaded_nl, just_loaded_nl_enid, just_loaded_nl_fft]
        ### current link end is destination
        if agent_cl_enid == getattr(agent, 'destin_nid'):
            return ['vl_sink', agent_cl_enid, 'sink_node', 0.1, 1e8, None, None, None]
        agent_nl_enid = getattr(agent, 'nl_enid') #agent_routes[agent_id][agent_cl_enid]
        agent_nl = getattr(agent, 'next_link') #self.link_nid_dict[(agent_cl_enid, agent_nl_enid)]
        agent_nl_fft = self.link_fft_dict[agent_nl]
        agent_nl_lanes = self.link_lanes_dict[agent_nl]
        ### next link end is destination, next next link does not exist
        if agent_nl_enid == getattr(agent, 'destin_nid'):
            return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, agent_nl_lanes, 'vl_sink', 'sink_node', 1e8]
        agent_nnl_enid = agent_routes[agent_id][agent_nl_enid]
        agent_nnl = self.link_nid_dict[(agent_nl_enid, agent_nnl_enid)]
        agent_nnl_lanes = self.link_lanes_dict[agent_nnl]
        return [agent_nl, agent_cl_enid, agent_nl_enid, agent_nl_fft, agent_nl_lanes, agent_nnl, agent_nnl_enid, agent_nnl_lanes]

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
    
    def run_node_model(self, agents=None, t=None):
        
        agents_df = agents.agents
        agent_routes = agents.agent_routes
        
        ### queue vehicles
        queue_agents_df = agents_df[
            agents_df['agent_status']==2].copy().reset_index(drop=True)
        if queue_agents_df.shape[0] == 0:
            ### no queuing agents 
            return agents_df

        ### determine which queued vehicles are at the front
        links_to_use_df = self.links.loc[(self.links['link_type']=='real') & (self.links['link_id'].isin(queue_agents_df['current_link']) | self.links['link_id'].isin(queue_agents_df['next_link']))]
        ### storage_remain
        links_storage_remain_dict = dict(zip(links_to_use_df['link_id'], links_to_use_df['storage_remain']))
        ### capacity, random each time
        capacity_array = np.floor(links_to_use_df['capacity'] + np.random.random(size=links_to_use_df.shape[0]))
        links_capacity_dict = dict(zip(links_to_use_df['link_id'], capacity_array))
        
        ### no constraints if not in dictionary; meaning the links are either virtual source or sink
        ### storage capacity and flow capacity are updated every time step
        ### other attributes that are not updated every step are obtained through find_next_link(), e.g., lanes, fft.
        queue_agents_df['nl_storage_remain'] = queue_agents_df['next_link'].map(links_storage_remain_dict).fillna(1e5)
        queue_agents_df['cl_out_capacity'] = queue_agents_df['current_link'].map(links_capacity_dict).fillna(1e5)
        queue_agents_df['nl_in_capacity'] = queue_agents_df['next_link'].map(links_capacity_dict).fillna(1e5)
        
        ### filter for front agents by output capacity of the current link 
        queue_agents_df['current_link_queue_time'] = t - queue_agents_df['cl_fft'] - queue_agents_df['current_link_enter_time']
        queue_agents_df['cl_position'] = queue_agents_df.sort_values(by='current_link_queue_time', ascending=True).groupby('current_link', sort=False).cumcount()
        ### filter for front agents by input capacity of the next link
        queue_agents_df['nl_position'] = queue_agents_df.sort_values(by='current_link_queue_time', ascending=True).groupby('next_link', sort=False).cumcount()
        ### really at the front
        queue_agents_df = queue_agents_df.loc[
            (queue_agents_df['cl_position']<queue_agents_df['cl_lanes']) 
            & (queue_agents_df['cl_position']<queue_agents_df['cl_out_capacity'])
            & (queue_agents_df['nl_position']<queue_agents_df['nl_lanes'])
            & (queue_agents_df['nl_position']<queue_agents_df['nl_in_capacity'])
            & (queue_agents_df['nl_position']*8 < queue_agents_df['nl_storage_remain'])
            ].copy()
        if queue_agents_df.shape[0] == 0:
            return agents_df

        ### go vehcles: primary vehicles (first arrive at stop sign) plus non conflict
        go_agents_df = queue_agents_df.copy()
        go_agents_df = go_agents_df.sort_values(by='current_link_enter_time', ascending=True)
        go_agents_df['primary_cl'] = go_agents_df.groupby('cl_enid', sort=False)['current_link'].transform('first')
        go_agents_df['primary_nl'] = go_agents_df.groupby('cl_enid', sort=False)['next_link'].transform('first')
        ### label non-conflict movements
        go_agents_df['no_conflict'] = list(map(self.no_conflict_moves.get, go_agents_df[['primary_cl', 'primary_nl', 'current_link', 'next_link']].to_records(index=False).tolist()))
        ### Cming from virtual source or going to virtual sink. Assume no conflict with any direction.
        go_agents_df['no_conflict'] = np.where(
            (go_agents_df['cl_enid']==go_agents_df['origin_nid']) | 
            (go_agents_df['cl_enid']==go_agents_df['destin_nid']), 1, go_agents_df['no_conflict'])
        ### no conflict if has the same movement as the primary movement
        go_agents_df = go_agents_df.loc[go_agents_df['no_conflict']==1]
        go_ids = go_agents_df['agent_id']
        if go_agents_df.shape[0] == 0:
            return agents_df

        ### update agent position
        agent_clnl_update = {getattr(agent, 'agent_id'): self.find_next_link(agent, agent_routes) for agent in go_agents_df.itertuples()}
        agents_df.loc[agents_df['agent_id'].isin(go_ids), ['current_link', 'cl_snid', 'cl_enid', 'cl_fft', 'cl_lanes', 'next_link', 'nl_enid', 'nl_lanes']] = pd.DataFrame(agents_df.loc[agents_df['agent_id'].isin(go_ids), 'agent_id'].map(agent_clnl_update).to_list()).values
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
            sp.clear()
        return route

    def update_route_info(self, reroute_agents, network, just_loaded=False):

        unarrived_list = []
        agent_clnl_update = dict()
        for agent in reroute_agents.itertuples():
            ### agent id
            agent_id = getattr(agent, 'agent_id')
            ### get initial or new routes
            route = self.get_agent_routes(getattr(agent, 'cl_enid'), getattr(agent, 'destin_nid'), network.g)
            ### update route
            self.agent_routes[agent_id] = route
            ### no path
            if len(route)==0: unarrived_list.append(agent_id)
            else: agent_clnl_update[agent_id] = network.find_next_link(agent, self.agent_routes, just_loaded=True)
        if len(unarrived_list)>0:
            self.agents.loc[self.agents['agent_id'].isin(unarrived_list), 'agent_status'] = -2
        self.agents.loc[self.agents['agent_id'].isin(
            reroute_agents['agent_id']), 
                        ['current_link', 'cl_snid', 'cl_enid', 'cl_fft', 'cl_lanes', 'next_link', 'nl_enid', 'nl_lanes'
                        ]] = pd.DataFrame(reroute_agents['agent_id'].map(agent_clnl_update).to_list()).values

    def load_agents(self, network=None, t=None):
        ### load agent
        load_ids = t == self.agents['departure_time']
        if np.sum(load_ids) == 0:
            return
        self.agents['cl_enid'] = self.agents['origin_nid']
        ### loaded to the queue on virtual source links
        self.agents['agent_status'] = np.where(load_ids, 2, self.agents['agent_status'])
        self.agents['current_link_enter_time'] = np.where(load_ids, t, self.agents['current_link_enter_time'])
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
    

    def run_one_step(self, t, reroute_frequency=None):
        
        ### load agents that are ready to departure; find routes and next links for them
        self.agents.load_agents(network=self.network, t=t)
        
        ### reroute agents
        self.agents.reroutes(network=self.network, t=t, reroute_frequency=reroute_frequency)
        self.agents.agents = self.network.run_link_model(agents=self.agents, t=t)
        self.agents.agents = self.network.run_node_model(agents=self.agents, t=t)




