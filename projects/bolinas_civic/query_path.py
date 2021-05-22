import os
import sys
import json
import pandas as pd

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, abs_path)
from sp import interface

def get_route(g, origin, destin):
    sp = g.dijkstra(origin, destin)
    sp_dist = sp.distance(destin)
    if (sp_dist>1e8):
        sp.clear()
        print('no path found')
        return {}
    else:
        sp_route = sp.route(destin)
        route = {}
        for (start_nid, end_nid) in sp_route:
            route[start_nid] = end_nid
        sp.clear()
        return route

def query_path(vphh=None, visitor_cnts=None, player_origin=None, player_destin=None, start_time=None, end_time=None, read_path = None):

    if read_path == None:
        read_path = abs_path

    ### get graph
    links_df = pd.read_csv(read_path + '/simulation_outputs/network/modified_network_edges_vphh{}_visitor{}.csv'.format(vphh, visitor_cnts))
    network_g = interface.from_dataframe(links_df, 'nid_s', 'nid_e', 'fft')

    network_links = json.load(open(read_path + '/simulation_outputs/network/network_links.json'))
    node2link_dict = json.load(open(read_path + '/simulation_outputs/network/node2link_dict.json'))

    link_speed_dict = json.load(open(read_path + '/simulation_outputs/link_weights/link_speed_vphh{}_visitor{}.json'.format(vphh, visitor_cnts)))
    link_length_dict = {getattr(l, 'eid'): getattr(l, 'length') for l in links_df.itertuples()}
    
    current_link = 'n{}_vl'.format(player_origin)
    current_link_distance = 0
    # current_link_angle = 0

    # all nodes that player passed
    player_nodes = [player_origin]
    player_nodes_time_traffic = []
    
    for t_p in range(start_time, end_time):
        if t_p > start_time:
            current_link_distance += link_speed_dict[str(t_p)][current_link]
        
        if (t_p == start_time) or (current_link_distance >= network_links[str(current_link)]['length']):
            
            if network_links[str(current_link)]['end_nid'] == player_destin:
                print('Reach destination at {} seconds'.format(t_p))
                break
            
            # extra distance to next link
            current_link_distance = current_link_distance - network_links[str(current_link)]['length']
            
            # reroute at intersection
            links_df['current_travel_time'] = link_speed_dict[str(t_p)]
            network_g = interface.from_dataframe(links_df, 'nid_s', 'nid_e', 'current_travel_time')
            player_route_by_nodes = get_route(network_g, network_links[str(current_link)]['end_nid'], player_destin)
            # print(player_route_by_nodes)
            
            ### move agent to to chosen link
            next_node = player_route_by_nodes[network_links[str(current_link)]['end_nid']]
            current_link = node2link_dict[str(network_links[str(current_link)]['end_nid'])][str(next_node)]
            player_nodes.append(network_links[str(current_link)]['end_nid'])
            player_nodes_time_traffic.append((network_links[str(current_link)]['start_nid'], t_p, link_speed_dict[str(t_p)][current_link], link_length_dict[current_link]))
            # print('new link {} at {}\n'.format(current_link, t_p))  
    
    print('vehicle is on link {} at {} seconds. The end node ID of the current link is {}'.format(current_link, end_time, next_node))
    print('Player path nodes {}'.format(player_nodes))
    print('Player path nodes/time/speed_of_next_link/length_of_next_link {}'.format(player_nodes_time_traffic))
    # traffic: light: > 9 m/s, medium: 3-9 m/s, heavy: < 3 m/s

    return player_nodes, player_nodes_time_traffic