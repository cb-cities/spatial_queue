import sys
import random
import numpy as np 
from ctypes import c_double
import scipy.io as sio
import scipy.sparse as ssparse

sys.path.insert(0, '/home/bingyu/Documents')
from sp import interface 

from .node import Node
from .link import Link

class Network:
    
    def __init__(self, nodes_df, links_df):

        self.nodes = dict()
        self.links = dict()
        self.nodes_osmid_dict = dict()
        self.node2link_dict = dict()
        self.g = None

        for n in nodes_df.itertuples():
            ### real node
            self.nodes[getattr(n, 'node_id_igraph')] = Node(nid=getattr(n, 'node_id_igraph'), lon=getattr(n, 'lon'), lat=getattr(n, 'lat'), ntype='real', osmid=getattr(n, 'node_osmid'))
            ### virtual node
            self.nodes['vn{}'.format(getattr(n, 'node_id_igraph'))] = self.nodes[getattr(n, 'node_id_igraph')].create_virtual_node()
            ### virtual link
            self.links['n{}_vl'.format(getattr(n, 'node_id_igraph'))] = self.nodes[getattr(n, 'node_id_igraph')].create_virtual_link()
            ### osmid dict
            self.nodes_osmid_dict[getattr(n, 'node_osmid')] = getattr(n, 'node_id_igraph')
       
        for l in links_df.itertuples():
            ### real link
            self.links[getattr(l, 'edge_id_igraph')] = Link(getattr(l, 'edge_id_igraph'), getattr(l, 'lanes'), getattr(l, 'length'), getattr(l, 'fft'), getattr(l, 'capacity'), 'real', getattr(l, 'start_igraph'), getattr(l, 'end_igraph'), getattr(l, 'geometry'))
            
        for n in self.nodes.values(): self.__calculate_straight_ahead_links(n)
        ### add in links and out links for end node, node-to-link dictionary
        for l in self.links.values(): 
            self.nodes[l.start_nid].out_links.append(l.id)
            self.nodes[l.end_nid].in_links[l.id] = None
            self.node2link_dict[(l.start_nid, l.end_nid)] = l.id
        
        ### Convert to mtx
        wgh = links_df['fft']
        row = links_df['start_igraph']
        col = links_df['end_igraph']
        assert max(np.max(row)+1, np.max(col)+1) == nodes_df.shape[0], 'nodes and links dimension do not match, row {}, col {}, nodes {}'.format(np.max(row), np.max(col), nodes_df.shape[0])
        g_coo = ssparse.coo_matrix((wgh, (row, col)), shape=(nodes_df.shape[0], nodes_df.shape[0]))
        sio.mmwrite('network_sparse.mtx', g_coo)
        # g_coo = sio.mmread(absolute_path+'/outputs/network_sparse.mtx'.format(folder))
        self.g = interface.readgraph(bytes('network_sparse.mtx', encoding='utf-8'))
    
    def one_step(self, t_now=None, agents=None):
        for l in self.links.values(): l.run_link_model(t_now=t_now, agents=agents)
        for n in self.nodes.values(): 
            n_t_move, n_t_key_loc_flow = self.__run_node_model(t_now=t_now, n=n, agents=agents)
        return n_t_move, n_t_key_loc_flow
    
    def update_all_links_travel_time(self, t_now=None, link_time_lookback_freq=None):
        for link in self.links.values(): 
            link_s, link_e, link_new_weight = link.update_travel_time(t_now, link_time_lookback_freq)
            if link_new_weight is not None:
                self.g.update_edge(link_s, link_e, c_double(link_new_weight))

    def __calculate_straight_ahead_links(self, n):
        for il in n.in_links.keys():
            x_start = self.nodes[self.links[il].start_nid].lon
            y_start = self.nodes[self.links[il].start_nid].lat
            in_vec = (n.lon-x_start, n.lat-y_start)
            sa_ol = None ### straight ahead out link
            ol_dir = 180
            for ol in n.out_links:
                x_end = self.nodes[self.links[ol].end_nid].lon
                y_end = self.nodes[self.links[ol].end_nid].lat
                out_vec = (x_end-n.lon, y_end-n.lat)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                new_ol_dir = np.arctan2(det, dot)*180/np.pi
                if abs(new_ol_dir)<ol_dir:
                    sa_ol = ol
                    ol_dir = new_ol_dir
            if (abs(ol_dir)<=45) and self.links[il].ltype=='real':
                n.in_links[il] = sa_ol

    def __find_go_vehs(self, n, go_link, agents=None):
        go_vehs_list = []
        incoming_lanes = int(np.floor(go_link.lanes))
        incoming_vehs = len(go_link.queue_veh)
        for ln in range(min(incoming_lanes, incoming_vehs)):
            agent_id = go_link.queue_veh[ln]
            agent_next_node, ol, agent_dir = agents[agent_id].prepare_agent(node_id = n.id)   
            go_vehs_list.append([agent_id, agent_next_node, go_link.id, ol, agent_dir])
        return go_vehs_list

    def __non_conflict_vehs(self, n, agents=None):
        n.go_vehs = []
        ### a primary direction
        in_links = [l for l in n.in_links.keys() if len(self.links[l].queue_veh)>0]
        if len(in_links) == 0: return
        go_link = self.links[random.choice(in_links)]
        go_vehs_list = self.__find_go_vehs(n, go_link, agents=agents)
        n.go_vehs += go_vehs_list
        ### a non-conflicting direction
        if (np.min([veh[-1] for veh in go_vehs_list])<-45) or (go_link.ltype=='v'): return ### no opposite veh allows to move if there is left turn veh in the primary direction; or if the primary incoming link is a virtual link
        if n.in_links[go_link.id] == None: return ### no straight ahead opposite links
        op_go_link = self.links[n.in_links[go_link.id]]
        op_go_link = self.links[self.node2link_dict[(op_go_link.end_nid, op_go_link.start_nid)]]
        op_go_vehs_list = self.__find_go_vehs(n, op_go_link, agents=agents)
        n.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side

    def __run_node_model(self, t_now=None, n=None, agents=None):
        self.__non_conflict_vehs(n, agents=agents)
        node_move = 0
        n_t_key_loc_flow = 0
        ### Agent reaching destination
        for [agent_id, next_node, il, ol, agent_dir] in n.go_vehs:
            ### arrival
            if (next_node is None) and (n.id == agents[agent_id].destin_nid):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                self.links[il].send_veh(t_now, agents[agent_id])
                n_t_key_loc_flow += agents[agent_id].move_agent(t_now, n.id, next_node, 'arr', il, ol)
            ### no storage capacity downstream
            elif self.links[ol].st_c < 1:
                pass ### no blocking, as # veh = # lanes
            ### inlink-sending, outlink-receiving both permits
            elif (self.links[il].ou_c >= 1) & (self.links[ol].in_c >= 1):
                node_move += 1
                ### before move agent as it uses the old agent.cl_enter_time
                self.links[il].send_veh(t_now, agents[agent_id])
                n_t_key_loc_flow += agents[agent_id].move_agent(t_now, n.id, next_node, 'flow', il, ol)
                self.links[ol].receive_veh(agent_id)
            ### either inlink-sending or outlink-receiving or both exhaust
            else:
                control_cap = min(self.links[il].ou_c, self.links[ol].in_c)
                toss_coin = random.choices([0,1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]:
                    node_move += 1
                    ### before move agent as it uses the old agent.cl_enter_time
                    self.links[il].send_veh(t_now, agents[agent_id])
                    n_t_key_loc_flow += agents[agent_id].move_agent(t_now, n.id, next_node, 'chance', il, ol)
                    self.links[ol].receive_veh(agent_id)
                else:
                    pass
        return node_move, n_t_key_loc_flow