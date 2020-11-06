from keplergl import KeplerGl
import model.dta_meso_butte as dta_meso

import json
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads

import random
import numpy as np
random.seed(0)
np.random.seed(0)

def extract_vehicle_locations(network):
    agents_dict = network.agents.copy()
    links_dict = network.links.copy()
    queue_vehicle_position = []
    run_vehicle_position = []
    closed_link_position = []
    for link_id, link in links_dict.items():
        if link.link_type == 'v':
            continue
        if link.burnt == 'burnt':
            closed_link_position.append([link.link_id, link.midpoint[0], link.midpoint[1]])
        link_geometry = link.geometry
        link_length = link.geometry.length
        link_lanes = link.lanes
        queue_vehicle_i, run_vehicle_i = 0, 0
        for vehicle in link.queue_vehicles:
            vehicle_pos = link_geometry.interpolate( link_length - queue_vehicle_i*8/link_lanes )
            queue_vehicle_position.append([vehicle, vehicle_pos.x, vehicle_pos.y])
            queue_vehicle_i += 1
        for vehicle in link.run_vehicles:
            vehicle_pos = link_geometry.interpolate( run_vehicle_i*8/link_lanes )
            run_vehicle_position.append([vehicle, vehicle_pos.x, vehicle_pos.y])
            run_vehicle_i += 1
    return run_vehicle_position, queue_vehicle_position, closed_link_position

def make_gdf(vehicle_position_list):
    vehicle_position_df = pd.DataFrame(vehicle_position_list, columns=['vehicle_id', 'x', 'y'])
    vehicle_position_gdf = gpd.GeoDataFrame(vehicle_position_df, crs='epsg:26910', geometry=[Point(xy) for xy in zip(vehicle_position_df['x'], vehicle_position_df['y'])]).to_crs('epsg:4326')
    vehicle_position_gdf['lon'] = vehicle_position_gdf['geometry'].apply(lambda x: x.x)
    vehicle_position_gdf['lat'] = vehicle_position_gdf['geometry'].apply(lambda x: x.y)
    vehicle_position_gdf = vehicle_position_gdf[['vehicle_id', 'lon', 'lat']]
    return vehicle_position_gdf

def add_fire(map_vehicles, fire_df):
    fire_polygons = []
    for fire in fire_df.loc[fire_df['lon']<-120].itertuples():
        fire_start_time = getattr(fire, 'start_time')
        fire_end_time = getattr(fire, 'end_time')
        fire_speed = getattr(fire, 'speed')
        fire_type = getattr(fire, 'type')
        fire_origin = getattr(fire, 'geometry')
        fire_offset_dist = getattr(fire, 'offset')
        if fire_start_time > 2000:
            fire_polygons.append([fire_type, fire_origin.buffer(1)])
        elif fire_type != 'initial':
            fire_polygons.append([fire_type, fire_origin.buffer(fire_offset_dist)])
        else:
            fire_time = min(fire_end_time, max(fire_start_time, 2000)) - fire_start_time
            fire_polygons.append([fire_type, fire_origin.buffer(fire_speed * fire_time)])
    fire_polygons_df = pd.DataFrame(fire_polygons, columns=['fire_type', 'geometry'])
    fire_polygons_gdf = gpd.GeoDataFrame(fire_polygons_df, crs='epsg:26910', geometry=fire_polygons_df['geometry']).to_crs('epsg:4326')
    
    map_vehicles.add_data(data=fire_polygons_gdf, name='fire_polygons')
    return map_vehicles

def add_closed_links(closed_link_position):
    closed_link_df = pd.DataFrame(closed_link_position, columns=['link_id', 'x', 'y'])
    closed_link_gdf = gpd.GeoDataFrame(closed_link_df, crs='epsg:26910', geometry=[Point(xy) for xy in zip(closed_link_df['x'], closed_link_df['y'])]).to_crs('epsg:4326')
    closed_link_gdf['lon'] = closed_link_gdf['geometry'].apply(lambda x: x.x)
    closed_link_gdf['lat'] = closed_link_gdf['geometry'].apply(lambda x: x.y)
    closed_link_gdf = closed_link_gdf[['link_id', 'lon', 'lat']]
    return closed_link_gdf

def make_map(network):
    run_vehicle_position, queue_vehicle_position, closed_link_position = extract_vehicle_locations(network)
    # make gdf
    queue_vehicle_position_gdf = make_gdf(queue_vehicle_position)
    run_vehicle_position_gdf = make_gdf(run_vehicle_position)
    closed_link_gdf = add_closed_links(closed_link_position)
    # make map
    with open('keplergl_config.json') as jsonfile:
        map_config = json.load(jsonfile)
    map_vehicles = KeplerGl(height=400, config=map_config)
    # map_vehicles = KeplerGl(height=400)
    map_vehicles.add_data(data=queue_vehicle_position_gdf, name='queue vehicles')
    map_vehicles.add_data(data=run_vehicle_position_gdf, name='run vehicles')
    map_vehicles.add_data(data=closed_link_gdf, name='closed_links')
    return map_vehicles

def main(vphh_id=None, dept_id=None, clos_id=None, contra_id=None, rout_id=None):
    # preparation
    scen_nm = "no_fire_v{}_d{}_cl{}_ct{}_r{}".format(vphh_id, dept_id, clos_id, contra_id, rout_id)
    data, config = dta_meso.preparation(random_seed=0, vphh_id=vphh_id, dept_id=dept_id, clos_id=clos_id, contra_id=contra_id, rout_id=rout_id, scen_nm=scen_nm)

    fitness=0
    visualization_t_list = {200, 800, 2000, 4000, 6000, 8000, 12000, 14000, 16000, 18000}
    for t in range(0, 20001):
        step_fitness, network = dta_meso.one_step(t, data, config)
        if step_fitness is not None:
            fitness += step_fitness
        if t in visualization_t_list:
            # visualization_t_dict[t] = make_map(network)
            map = make_map(network)
            map.save_to_html(file_name="projects/butte_osmnx/visualization_outputs/map_{}_{}.html".format(scen_nm, t))

if __name__ == "__main__":
    main(vphh_id='123', dept_id='2', rout_id='2')