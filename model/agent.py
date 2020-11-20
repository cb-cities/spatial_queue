
class Agent:
    def __init__(self, agent_id, origin_nid, destin_nid, deptarture_time, vehicle_length, gps_reroute):
        #input
        self.agent_id = agent_id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.furthest_nid = destin_nid
        self.deptarture_time = deptarture_time
        self.vehicle_length = vehicle_length
        self.gps_reroute = gps_reroute
        ### derived
        self.current_link_start_nid = 'vn{}'.format(self.origin_nid) # current link start node
        self.current_link_end_nid = self.origin_nid # current link end node
        ### Empty
        self.route = {} ### None or list of route
        self.status = 'unloaded' ### unloaded, enroute, shelter, shelter_arrive, arrive
        self.current_link_enter_time = None
        self.next_link = None
        self.next_link_end_nid = None ### next link end

    def get_path():
        sp = g.dijkstra(self.current_link_end_nid, self.destin_nid)
        sp_dist = sp.distance(self.destin_nid)
        if sp_dist > 1e8:
            sp.clear()
            # self.route = {self.current_link_start_nid: self.current_link_end_nid}
            self.route = {}
            self.furthest_nid = self.current_link_end_nid
            self.status = 'shelter'
            print(self.agent_id, self.current_link_start_nid, self.current_link_end_nid)
        else:
            sp_route = sp.route(self.destin_nid)
            ### create a path. Order only preserved in Python 3.7+. Do not rely on.
            # self.route = {self.current_link_start_nid: self.current_link_end_nid}
            for (start_nid, end_nid) in sp_route:
                self.route[start_nid] = end_nid
            sp.clear()

node_edge_df.to_csv('/home/bingyu/Documents/spatial_queue/projects/butte_osmnx/simulation_outputs/network/node_edge_df.csv', index=False)
