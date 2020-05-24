#! Python 3
import numpy as np
import shapely.vectorized as sv 

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    From: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def point_to_vertex_dist_positive(veh_lon, veh_lat, vertex_list):
    """
    Calculate the min distance between a list of points and a vector's vertex
    Use the point-vertex distance to represent the point-polygon distance
    """
    dist = np.full(len(veh_lon), np.inf)
    for (vertex_lon, vertex_lat) in vertex_list:
        vertex_dist = haversine(veh_lat, veh_lon, vertex_lat, vertex_lon)
        dist = np.minimum(dist, vertex_dist)
    return dist

def point_to_vertex_dist(veh_lon, veh_lat, polygon_geom):
    """
    Put a negative sign on the point contained by polygon
    https://github.com/geopandas/geopandas/issues/430#issuecomment-291003750
    """
    contain_array = sv.contains(polygon_geom, veh_lon, veh_lat) ### vectorized contain
    contain_array = np.where(contain_array, -1, 1) ### translate contain to -1, not contain to 1
    distance_array = point_to_vertex_dist_positive(veh_lon, veh_lat, polygon_geom.exterior.coords)
    distance_array = distance_array * contain_array
    return distance_array