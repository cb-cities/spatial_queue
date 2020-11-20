import sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
from shapely.wkt import loads
from itertools import product

def interpolate_coordinates(row):
    road_geometry = getattr(row, 'geometry')
    (start_x, start_y) = road_geometry.coords[0]
    (start_ip_x, start_ip_y) = road_geometry.interpolate(0.1, normalized=True).coords[0]
    (end_x, end_y) = road_geometry.coords[-1]
    (end_ip_x, end_ip_y) = road_geometry.interpolate(0.9, normalized=True).coords[0]
    start_angle = np.arctan2(start_ip_y - start_y, start_ip_x - start_x)/np.pi*180
    end_angle = np.arctan2(end_y - end_ip_y, end_x - end_ip_x)/np.pi*180
    return pd.Series([start_angle, end_angle])

def network(nodes_file, edges_file, additional_file=None):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)
    edges['fft'] = edges['length']/edges['maxmph']*2.237
    edges = gpd.GeoDataFrame(edges, crs='epsg:4326', geometry=edges['geometry'].map(loads)).to_crs(26910)
    edges[['start_angle', 'end_angle']] = edges.apply(interpolate_coordinates, axis=1)
    print(edges.iloc[0])

    node_edge_dict = {getattr(node, 'nid'): {'incoming_links': [], 'outgoing_links': []} for node in nodes.itertuples()}
    for edge in edges.itertuples():
        node_edge_dict[getattr(edge, 'nid_s')]['outgoing_links'].append(getattr(edge, 'eid'))
        node_edge_dict[getattr(edge, 'nid_e')]['incoming_links'].append(getattr(edge, 'eid'))
    node_edge_list = []
    loading_edge_list = []
    leaving_edge_list = []
    for node_id, edge_info in node_edge_dict.items():
        for (incoming_link_id, outgoing_link_id) in list(product(edge_info['incoming_links'], edge_info['outgoing_links'])):
            node_edge_list.append([node_id, incoming_link_id, outgoing_link_id])
        for incoming_link_id in edge_info['incoming_links']:
            leaving_edge_list.append([node_id, incoming_link_id, 'leaving_{}'.format(node_id)])
        for outgoing_link_id in edge_info['outgoing_links']:
            loading_edge_list.append([node_id, 'loading_{}'.format(node_id), outgoing_link_id])

    node_edge_df = pd.DataFrame(node_edge_list, columns = ['nid', 'in_eid', 'out_eid'])
    node_edge_df = pd.merge(node_edge_df, edges[['eid', 'end_angle', 'type', 'fft']], left_on='in_eid', right_on='eid', how='left').drop(columns='eid')
    node_edge_df = pd.merge(node_edge_df, edges[['eid', 'start_angle', 'type', 'fft']], left_on='out_eid', right_on='eid', how='left', suffixes=['_in', '_out']).drop(columns='eid')
    node_edge_df['turning_angle'] = node_edge_df['end_angle'] - node_edge_df['start_angle']
    node_edge_df['turning_angle'] = np.where(node_edge_df['turning_angle']>180, 360-node_edge_df['turning_angle'], node_edge_df['turning_angle'])
    node_edge_df['turning_angle'] = np.where(node_edge_df['turning_angle']<-180, 360+node_edge_df['turning_angle'], node_edge_df['turning_angle'])
    node_edge_df = node_edge_df[['nid', 'in_eid', 'out_eid', 'turning_angle', 'type_in', 'type_out', 'fft_in', 'fft_out']]

    loading_leaving_edge_df = pd.DataFrame(loading_edge_list+leaving_edge_list, columns=['nid', 'in_eid', 'out_eid'])
    loading_leaving_edge_df = pd.merge(loading_leaving_edge_df, edges[['eid', 'type', 'fft']], left_on='in_eid', right_on='eid', how='left').drop(columns='eid')
    loading_leaving_edge_df = pd.merge(loading_leaving_edge_df, edges[['eid', 'type', 'fft']], left_on='out_eid', right_on='eid', how='left', suffixes=['_in', '_out']).drop(columns='eid')
    loading_leaving_edge_df = loading_leaving_edge_df.fillna(value={'type_in': 'virtual', 'type_out': 'virtual', 'fft_in': 0, 'fft_out': 0})
    loading_leaving_edge_df['turning_angle'] = 0.0
    loading_leaving_edge_df = loading_leaving_edge_df[['nid', 'in_eid', 'out_eid', 'turning_angle', 'type_in', 'type_out', 'fft_in', 'fft_out']]

    print(node_edge_df.shape)
    print(node_edge_df.head(2))
    print(loading_leaving_edge_df.head(2))
    node_edge_df = pd.concat([node_edge_df, loading_leaving_edge_df])
    print(node_edge_df.shape)
    print(node_edge_df.head(2))
    node_edge_df.to_csv('/home/bingyu/Documents/spatial_queue/projects/butte_osmnx/simulation_outputs/network/node_edge_df.csv', index=False)


if __name__ == "__main__":
    nodes_file = '/home/bingyu/Documents/spatial_queue/projects/butte_osmnx/network_inputs/butte_nodes_sim.csv'
    edges_file = '/home/bingyu/Documents/spatial_queue/projects/butte_osmnx/network_inputs/butte_edges_sim.csv'
    network(nodes_file, edges_file)