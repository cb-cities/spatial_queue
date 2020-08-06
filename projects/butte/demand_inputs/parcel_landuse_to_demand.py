import os 
import sys 
import random 
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import shapely
from shapely import wkt, ops 
import shapely.speedups
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist 

shapely.speedups.enable()

absolute_path = os.path.dirname(os.path.abspath(__file__))

def parcels():

    ### read city/town boundary files
    cities = gpd.read_file(absolute_path+'/../network_inputs/ca-places-boundaries/CA_Places_TIGER2016.shp').to_crs({'init': 'epsg:3857'})
    evacuation_area = cities.loc[cities['NAME'].isin(['Paradise', 'Magalia'])].unary_union
    print('evacuation area {}km**2'.format(evacuation_area.area/1e6))

    ### read nodes file
    nodes_df = pd.read_csv(absolute_path+'/../network_inputs/osm_nodes.csv')
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=[shapely.geometry.Point(xy) for xy in zip(nodes_df.lon, nodes_df.lat)], crs={'init': 'epsg:4326'}).to_crs({'init': 'epsg:3857'})

    ### Land use data requested from Butte County Association of Governments (BCAG)
    parcels = gpd.read_file(absolute_path+'/../network_inputs/BCAG_GIS_Request/BCAG_Land_Use/BCAG_Land_Use.shp').to_crs({'init': 'epsg:3857'})
    parcels['centroid'] = parcels.apply(lambda x: x['geometry'].centroid, axis=1)
    parcels = parcels.set_geometry('centroid')

    ### Based on https://geoffboeing.com/2016/10/r-tree-spatial-index-python/?share=google-plus-1
    parcels_sindex = parcels.sindex
    coarse_parcel_ids = list(parcels_sindex.intersection(evacuation_area.bounds))
    coarse_parcels = parcels.iloc[coarse_parcel_ids]
    precise_parcels = coarse_parcels[coarse_parcels.intersects(evacuation_area)]
    parcels_evac = parcels[parcels['APN'].isin(precise_parcels['APN'].values)].reset_index()
    print('numbers of parcels in evacuation area (Paradise, Magalia) {}, {} residential units'.format(parcels_evac.shape[0], np.sum(parcels_evac['RES_UNITS'])))

    ### assign closest node
    parcels_evac['c_x'] = parcels_evac['centroid'].x
    parcels_evac['c_y'] = parcels_evac['centroid'].y
    nodes_xy = np.array([nodes_gdf['geometry'].x.values, nodes_gdf['geometry'].y.values]).transpose()
    nodes_osmid = nodes_gdf['node_osmid'].values
    def get_closest_node(parcel_x, parcel_y):
        return nodes_osmid[cdist([(parcel_x, parcel_y)], nodes_xy).argmin()]
    parcels_evac['closest_node'] = parcels_evac.apply(lambda x: get_closest_node(x['c_x'], x['c_y']), axis=1)
    print(parcels_evac.iloc[0])
    # parcels_evac.to_csv(absolute_path+'/land_use_parcels_paradise_magalia.csv', index=False)
    
    ### safe nodes
    ### all nodes in Chico and Orovile
    chico_area = cities.loc[cities['NAME'].isin(['Chico'])]
    oroville_area = cities.loc[cities['NAME'].isin(['Palermo', 'Oroville East', 'South Oroville', 'Kelly Ridge', 'Oroville', 'Thermalito'])]
    print(oroville_area['NAME'].values.tolist()) ### Should be 6
    safe_area_dict = {'chico': chico_area.unary_union, 'oroville': oroville_area.unary_union}
    safe_nodes = dict()
    nodes_sindex = nodes_gdf.sindex
    for nm, geom in safe_area_dict.items():
        coarse_node_ids = list(nodes_sindex.intersection(geom.bounds))
        coarse_nodes = nodes_gdf.iloc[coarse_node_ids]
        precise_nodes = coarse_nodes[coarse_nodes.intersects(geom)]
        safe_nodes[nm] = precise_nodes['node_osmid'].values.tolist()
        print(len(precise_nodes))

    chico_ratio = 0.7
    o_list = [getattr(p, 'closest_node') for p in parcels_evac.itertuples() for r in range(int(getattr(p, 'RES_UNITS')))]
    d_list = random.choices(safe_nodes['chico'], k=int(len(o_list)*chico_ratio))
    d_list += random.choices(safe_nodes['oroville'], k=len(o_list)-len(d_list))
    od = pd.DataFrame({'o_osmid': o_list, 'd_osmid': d_list})
    od = pd.merge(od, nodes_df[['node_osmid', 'lon', 'lat']], how='left', left_on='o_osmid', right_on='node_osmid')
    od = pd.merge(od, nodes_df[['node_osmid', 'lon', 'lat']], how='left', left_on='d_osmid', right_on='node_osmid', suffixes=['_start', '_end'])

    od[['o_osmid', 'd_osmid', 'lon_start', 'lat_start', 'lon_end', 'lat_end']].to_csv(absolute_path+'/od_a{}.csv'.format(od.shape[0]), index=False)


def main():
    parcels()

if __name__ == '__main__':
    main()