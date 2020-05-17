#! Python 3
import numpy as np

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

def point_to_vertex_dist(veh_lon, veh_lat, vertex_list):
    """
    Calculate the min distance between a list of points and a vector's vertex
    Use the point-vertex distance to represent the point-polygon distance
    """
    dist = np.full(len(veh_lon), np.inf)
    for (vertex_lon, vertex_lat) in vertex_list:
        vertex_dist = haversine(veh_lat, veh_lon, vertex_lat, vertex_lon)
        dist = np.minimum(dist, vertex_dist)
    return dist