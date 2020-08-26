import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

### remove loops
def remove_loops(edges):
    edges_no_loop = edges[edges['u']!=edges['v']].copy()
    print('remove_loops() removes {} edges'.format(edges.shape[0] - edges_no_loop.shape[0]))
    return edges_no_loop

### remove multiedges
def remove_multiedges(edges):
    sorter=['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'residential', 'unclassified']
    edges['highway'] = edges['highway'].astype('category').cat.set_categories(sorter)
    edges_no_multiedges = edges.sort_values(['highway']).drop_duplicates(subset=['u', 'v'], keep='first')
    print('remove_multiedges() removes {} edges'.format(edges.shape[0] - edges_no_multiedges.shape[0]))
    return edges_no_multiedges

def undirected_edges(edges):
    ### create undirected edges
    edge_ends_undirected = edges[['u', 'v']].copy()
    edge_ends_undirected['large_uv'] = np.max(edge_ends_undirected[['u', 'v']], axis=1)
    edge_ends_undirected['small_uv'] = np.min(edge_ends_undirected[['u', 'v']], axis=1)
    edge_ends_undirected = edge_ends_undirected.drop_duplicates(subset=['large_uv', 'small_uv'], keep='first')
    node_cnts = pd.DataFrame({
        'node_osmid': edge_ends_undirected['large_uv'].values.tolist() + edge_ends_undirected['small_uv'].values.tolist()}).groupby('node_osmid').size().to_frame('cnt')
    return edge_ends_undirected, node_cnts

### remove staggering nodes
def find_stag_nodes(edges):
    edge_ends_undirected, node_cnts = undirected_edges(edges)
    nodes_stag = node_cnts[node_cnts['cnt']==1].reset_index()
    print('  # staggring nodes: ', nodes_stag.shape[0])
    return nodes_stag

### connect non-intersection nodes
def find_non_intersections(edges_remain):
    edge_ends_undirected, node_cnts = undirected_edges(edges_remain)
    nodes_non_intersection = node_cnts[node_cnts['cnt']==2].index.values.tolist()
    print('  # staggring nodes: {}, # non-intersection nodes {}'.format(node_cnts[node_cnts['cnt']==1].shape[0], len(nodes_non_intersection)))
    return nodes_non_intersection

def remove_stag_nodes(nodes, edges, iteration=5):
    edges = remove_loops(edges)
    edges = remove_multiedges(edges)

    # initialization
    nodes_no_stag = nodes.copy()
    edges_no_stag = edges.copy()
    for iteration in range(iteration):
        print('iteration ', iteration)
        stag_nodes = find_stag_nodes(edges_no_stag)
        nodes_no_stag = nodes_no_stag[~nodes_no_stag['osmid'].isin(stag_nodes['node_osmid'])]
        edges_no_stag = edges_no_stag[(~edges_no_stag['u'].isin(stag_nodes['node_osmid'])) & (~edges_no_stag['v'].isin(stag_nodes['node_osmid']))]
    return nodes_no_stag, edges_no_stag

def remove_false_intersections(nodes, edges):
    edges = remove_loops(edges)
    edges = remove_multiedges(edges)

    # initialization
    nodes_remain = nodes.copy()
    edges_remain = edges.copy()
    # find false intersections
    nodes_non_intersection = find_non_intersections(edges_remain)
    print('# false intersections: ', len(nodes_non_intersection))
    print('# edges before removing false intersections: ', edges_remain.shape)

    for false_intersection in nodes_non_intersection:
        affected_edges = edges_remain[(edges_remain['u']==false_intersection) | (edges_remain['v']==false_intersection)]
        not_affected_edges = edges_remain[(edges_remain['u']!=false_intersection) & (edges_remain['v']!=false_intersection)]
        affected_nodes = affected_edges['u'].values.tolist() + affected_edges['v'].values.tolist()
        affected_nodes = list(set([n for n in affected_nodes if n!=false_intersection]))
        if len(affected_nodes) != 2:
            # print(affected_nodes, ' affected nodes != 2')
            continue ### non-standard geometry   
        if affected_edges.shape[0] not in [2, 4]:
            # print(false_intersection, ' affected edges != 4 or 2, it is ', affected_edges.shape[0])
            # print(affected_edges)
            continue ### non-standard geometry 
        if len(affected_edges['highway'].unique()) != 1: 
            # print(false_intersection, list(affected_edges['highway'].unique()))
            continue ### different road type

        
        n1, n2 = affected_nodes[0], affected_nodes[1]
        new_edge_list = []
        try:
            n1_m = affected_edges[(affected_edges['u']==n1) & (affected_edges['v']==false_intersection)]
            m_n2 = affected_edges[(affected_edges['u']==false_intersection) & (affected_edges['v']==n2)]
            ### connect n1_m with m_n2
            n1_n2 = n1_m.copy()
            n1_n2['v'] = n2
            n1_n2['geometry'] = n1_m.iloc[0]['geometry'].split(')')[0] + ', ' + m_n2.iloc[0]['geometry'].split('(')[1]
            ### new edges dataframe
            new_edge_list.append(n1_n2)
        except IndexError:
            pass

        try:
            n2_m = affected_edges[(affected_edges['u']==n2) & (affected_edges['v']==false_intersection)]
            m_n1 = affected_edges[(affected_edges['u']==false_intersection) & (affected_edges['v']==n1)]
            ### connect n2_m with m_n1
            n2_n1 = n2_m.copy()
            n2_n1['v'] = n1
            n2_n1['geometry'] = n2_m.iloc[0]['geometry'].split(')')[0] + ', ' + m_n1.iloc[0]['geometry'].split('(')[1]
            ### new edges dataframe
            new_edge_list.append(n2_n1)
        except IndexError:
            pass

        edges_remain = pd.concat([not_affected_edges] + new_edge_list)
        
    print('# edges after removing false intersections: ', edges_remain.shape)
    return nodes_remain, edges_remain

def gather_short_elements(edges, keep_edges=[]):
    keep_edges = ['{}-{}'.format(u, v) for (u, v) in keep_edges]
    edges = gpd.GeoDataFrame(edges[['u', 'v', 'highway', 'geometry']], crs='epsg:4326', geometry=edges['geometry'].apply(loads)).to_crs('epsg:3857')
    edges['length'] = edges['geometry'].length
    short_edges = edges[edges['length']<200].reset_index(drop=True).to_crs('epsg:4326')
    short_edges['u-v'] = short_edges.apply(lambda x: '{}-{}'.format(x['u'], x['v']), axis=1)
    short_edges = short_edges[~short_edges['u-v'].isin(keep_edges)]
    long_edges = edges.merge(short_edges[['u', 'v']].drop_duplicates(), on=['u','v'], 
                   how='left', indicator=True)
    long_edges = long_edges[long_edges['_merge'] == 'left_only'].to_crs('epsg:4326')
    print(edges.shape, short_edges.shape, long_edges.shape)
    
    short_edge_group_id = 0
    visited_nodes = dict()
    for s_e in short_edges.itertuples():
        s_e_u, s_e_v = getattr(s_e, 'u'), getattr(s_e, 'v')
        if (s_e_u in visited_nodes.keys()) and (s_e_v not in visited_nodes.keys()):
            visited_nodes[s_e_v] = visited_nodes[s_e_u]
        elif (s_e_v in visited_nodes.keys()) and (s_e_u not in visited_nodes.keys()):
            visited_nodes[s_e_u] = visited_nodes[s_e_v]
        elif (s_e_v in visited_nodes.keys()) and (s_e_u in visited_nodes.keys()):
            if visited_nodes[s_e_u] == visited_nodes[s_e_v]:
                pass ### both in the same group
            else: ### two ends in different group
                keep_grp = visited_nodes[s_e_u]
                discard_grp = visited_nodes[s_e_v]
                for k, v in visited_nodes.items():
                    if v==discard_grp:
                        visited_nodes[k]=keep_grp
        ### both not grouped yet
        else:
            visited_nodes[s_e_u] = short_edge_group_id
            visited_nodes[s_e_v] = short_edge_group_id
            short_edge_group_id += 1
                
    short_edges['grp'] = short_edges['u'].map(visited_nodes)

    grp_coord = dict()
    for grp, s_e_grp in short_edges.groupby('grp'):
        s_e_grp_x, s_e_grp_y = np.mean(s_e_grp['geometry'].centroid.x), np.mean(s_e_grp['geometry'].centroid.y)
        grp_coord[grp] = (s_e_grp_x, s_e_grp_y, '{}_m'.format(s_e_grp['u'].iloc[0]))
    
    return long_edges, visited_nodes, grp_coord

def short_elements_to_nodes(nodes, edges, keep_edges=[]):
    
    long_edges, visited_nodes, grp_coord = gather_short_elements(edges, keep_edges=keep_edges)
    new_long_edges_list = []
    for l_e in long_edges.itertuples():
        l_e_u, l_e_v, l_e_highway, l_e_geometry = getattr(l_e, 'u'), getattr(l_e, 'v'), getattr(l_e, 'highway'), getattr(l_e, 'geometry')
        ### change start nodes
        if l_e_u in visited_nodes.keys():
            new_l_e_u_x, new_l_e_u_y, new_l_e_u_id = grp_coord[visited_nodes[l_e_u]]
        else:
            new_l_e_u_x, new_l_e_u_y, new_l_e_u_id = l_e_geometry.coords[0][0], l_e_geometry.coords[0][1], l_e_u
        ### change end nodes
        if l_e_v in visited_nodes.keys():
            new_l_e_v_x, new_l_e_v_y, new_l_e_v_id = grp_coord[visited_nodes[l_e_v]]
        else:
            new_l_e_v_x, new_l_e_v_y, new_l_e_v_id = l_e_geometry.coords[-1][0], l_e_geometry.coords[-1][1], l_e_v
        ### new geometry
        new_l_e_coords = [(new_l_e_u_x, new_l_e_u_y)] + l_e_geometry.coords[1:-1] + [(new_l_e_v_x, new_l_e_v_y)]
        new_l_e_geometry = 'LINESTRING ({})'.format(','.join(['{} {}'.format(x, y) for (x, y) in new_l_e_coords]))
        new_long_edges_list.append([new_l_e_u_id, new_l_e_v_id, l_e_highway, new_l_e_geometry])

    ### make new dataframe containing only long roads
    edges_long = pd.DataFrame(new_long_edges_list, columns=['u', 'v', 'highway', 'geometry']) 
    edges_long = remove_loops(edges_long)
    edges_long = remove_multiedges(edges_long)
    nodes_long = nodes[(nodes['osmid'].isin(edges_long['u'])) | (nodes['osmid'].isin(edges_long['v']))][['x', 'y', 'osmid']]
    nodes_long = pd.concat([nodes_long, pd.DataFrame([[k[0], k[1], k[2]] for k in grp_coord.values()], columns=['x', 'y', 'osmid'])])
    return nodes_long, edges_long



if __name__ == '__main__':
    test_nodes = pd.read_csv('network_inputs/butte_simplified_ctm_nodes_no_stag.csv')
    test_edges = pd.read_csv('network_inputs/butte_simplified_ctm_edges_no_stag.csv')
    # test_edges['u'] = test_edges['start_osmid']
    # test_edges['v'] = test_edges['end_osmid']
    # test_edges['highway'] = test_edges['type']
    test_nodes_remain, test_edges_remain = remove_false_intersections(test_nodes, test_edges)
    # print('********* ', test_edges_remain[test_edges_remain['osmid']=='538864403'].iloc[0])


