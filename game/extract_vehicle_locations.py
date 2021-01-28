import numpy as np 
import pandas as pd 
import geopandas as gpd

def veh_offset(x1, y1, x2, y2):
    # tangential slope approximation
    tangent = (x2-x1, y2-y1)
    perp = (y2-y1, -(x2-x1))
    mode = np.sqrt(perp[0]**2+perp[1]**2)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    delta_x = perp[0]/mode*1.75
    delta_y = perp[1]/mode*1.75

    return (mid_x + delta_x, mid_y + delta_y), (tangent[0]/mode, tangent[1]/mode)

def extract_vehicle_locations(network, t):
    links_dict = network.links.copy()
    queue_vehicle_position = []
    run_vehicle_position = []

    for link_id, link in links_dict.items():
        
        ### skip virtual links
        if link.link_type == 'v':
            continue

        ### link stats
        link_fft = link.fft
        link_geometry = link.geometry
        link_length = link.geometry.length

        ### queuing
        queue_end = link_length - 4
        for q_veh_id in link.queue_vehicles:
            q_veh_loc = queue_end
            queue_end = max(queue_end-8, 4)
            q_veh_coord_1 = link_geometry.interpolate(q_veh_loc/link_length-0.001, normalized=True)
            q_veh_coord_2 = link_geometry.interpolate(q_veh_loc/link_length+0.001, normalized=True)
            q_veh_coord_offset, q_veh_dir = veh_offset(q_veh_coord_1.x, q_veh_coord_1.y, q_veh_coord_2.x, q_veh_coord_2.y)
            queue_vehicle_position.append([q_veh_id, 'q', link_id, q_veh_coord_offset[0], q_veh_coord_offset[1], q_veh_dir[0], q_veh_dir[1]])

        ### running
        for r_veh_id in link.run_vehicles:
            r_veh_current_link_enter_time = network.agents[r_veh_id].current_link_enter_time
            if link_length*(t-r_veh_current_link_enter_time)/link_fft>queue_end:
                r_veh_loc = queue_end
                queue_end = max(queue_end-8, 0)
            else:
                r_veh_loc = link_length*(t-r_veh_current_link_enter_time)/link_fft
            r_veh_loc = max(r_veh_loc, 4)
            r_veh_coord_1 = link_geometry.interpolate(r_veh_loc/link_length-0.001, normalized=True)
            r_veh_coord_2 = link_geometry.interpolate(r_veh_loc/link_length+0.001, normalized=True)
            r_veh_coord_offset, r_veh_dir = veh_offset(r_veh_coord_1.x, r_veh_coord_1.y, r_veh_coord_2.x, r_veh_coord_2.y)
            run_vehicle_position.append([r_veh_id, 'r', link_id, r_veh_coord_offset[0], r_veh_coord_offset[1], r_veh_dir[0], r_veh_dir[1]])
    
    veh_df = pd.DataFrame(queue_vehicle_position + run_vehicle_position, columns=['veh_id', 'status', 'link_id', 'lon_offset_utm', 'lat_offset_utm', 'dir_x', 'dir_y'])
    veh_df['lon_offset_sumo'] = veh_df['lon_offset_utm']-525331.68
    veh_df['lat_offset_sumo'] = veh_df['lat_offset_utm']-4194202.74
    # print(veh_df.iloc[0])
    # print(veh_df['lon_offset_utm'].iloc[0], veh_df['lon_offset_utm'].iloc[0]-518570.38)
    # veh_df.to_csv(simulation_outputs+'/veh_loc_interpolated/veh_loc_t{}.csv'.format(t), index=False)
    veh_gdf = gpd.GeoDataFrame(veh_df, crs='epsg:32610', geometry=gpd.points_from_xy(veh_df.lon_offset_utm, veh_df.lat_offset_utm))
    # veh_gdf.loc[veh_gdf['veh_id']==game_veh_id, 'status'] = 'g'

    return veh_gdf