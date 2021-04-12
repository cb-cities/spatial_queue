import os
import shapely
import numpy as np 
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from scipy.spatial.distance import cdist

absolute_path = os.path.dirname(os.path.abspath(__file__))

def map_parcels_to_nodes():
    ### read bolinas nodes
    nodes_df = pd.read_csv(absolute_path + '/../network_inputs/osm_nodes.csv')
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=[shapely.geometry.Point(xy) for xy in zip(nodes_df.lon, nodes_df.lat)], crs='epsg:4326')
    # display(nodes_gdf.head(2))
    
    ### read visitor nodes
    visitor_nodes_df = pd.read_csv(absolute_path + '/../network_inputs/osm_nodes_visitor_parking.csv')
    visitor_nodes_gdf = gpd.GeoDataFrame(visitor_nodes_df, geometry=[shapely.geometry.Point(xy) for xy in zip(visitor_nodes_df.lon, visitor_nodes_df.lat)], crs='epsg:4326')
    # display(visitor_nodes_gdf.head(2))

    ### Get the OSMid of the closest node to each parcel
    bolinas_parcels = pd.read_csv(absolute_path + '/parcels/parcels_bolinas/parcels_bolinas.csv')
    household_parcels = bolinas_parcels[bolinas_parcels['UseCd']==11].reset_index() ### only UseCd 11 has household size>0 (=1)
    household_parcels = gpd.GeoDataFrame(household_parcels, crs='epsg:4326',
                                         geometry=household_parcels['WKT'].apply(loads))

    household_parcels['centroid'] = household_parcels.apply(lambda x: x['geometry'].centroid, axis=1)
    household_parcels = household_parcels.set_geometry('centroid')
    household_parcels['c_x'] = household_parcels['centroid'].x
    household_parcels['c_y'] = household_parcels['centroid'].y
    nodes_xy = np.array([nodes_gdf['geometry'].x.values, nodes_gdf['geometry'].y.values]).transpose()
    nodes_osmid = nodes_gdf['node_osmid'].values

    def get_closest_node(parcel_x, parcel_y):
        return nodes_osmid[cdist([(parcel_x, parcel_y)], nodes_xy).argmin()]

    household_parcels['closest_node'] = household_parcels.apply(lambda x: get_closest_node(x['c_x'], x['c_y']), axis=1)
    # display(household_parcels.head(2))
    # parcels_evac.to_csv(absolute_path+'/outputs/parcels_evac.csv', index=False)
    
    return household_parcels, visitor_nodes_gdf

def generate_simple_od(vphh=1, visitor_cnts=0, demand_write_path=None):
    
    household_parcels, visitor_nodes_gdf = map_parcels_to_nodes()
    
    ### local residents origin
    local_residents_od_df = household_parcels[['Parcel', 'closest_node']].copy().sample(frac=vphh, replace=True).reset_index(drop=True)
    local_residents_od_df['origin_osmid'] = local_residents_od_df['closest_node']
    local_residents_od_df = local_residents_od_df.drop('closest_node', 1)
    local_residents_od_df['dept_time'] = 0
    local_residents_od_df['type'] = 'local'

    ### visitor origin
    visitor_vehicle_origin = np.random.choice(visitor_nodes_gdf['node_osmid'], size=visitor_cnts)
    visitor_vehicle_od_df = pd.DataFrame({'origin_osmid': visitor_vehicle_origin})
    visitor_vehicle_od_df = visitor_vehicle_od_df.sample(frac=1).reset_index(drop=True)
    visitor_vehicle_od_df['dept_time'] = 0
    visitor_vehicle_od_df['type'] = 'visitor'
    visitor_vehicle_od_df['Parcel'] = None
    
    ### combine resident OD and visitor OD
    od_df = pd.concat([local_residents_od_df]+[visitor_vehicle_od_df])
    
    ### set destination
    od_df['destin_osmid'] = '110397253'
    #od_df['destin_osmid'] = np.random.choice(['110360959', '110397253'], size=od_df.shape[0])
    
    ### save to output
    od_df.to_csv(demand_write_path + '/od_csv/resident_visitor_od_vphh{}_visitor{}.csv'.format(vphh, visitor_cnts), index=False)

