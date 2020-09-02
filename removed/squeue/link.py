import sys
import numpy as np 

class Link:
    def __init__(self, lid, lanes, length, fft, capacity, ltype, start_nid, end_nid, geometry):
        ### input
        self.id = lid
        self.lanes = lanes
        self.length = length
        self.fft = fft
        self.capacity = capacity
        self.ltype = ltype
        self.start_nid = start_nid
        self.end_nid = end_nid
        self.geometry = geometry
        ### derived
        self.store_cap = max(1, length*lanes/8)
        self.in_c = self.capacity/3600.0 # capacity in veh/s
        self.ou_c = self.capacity/3600.0
        self.st_c = self.store_cap # remaining storage capacity
        ### empty
        self.queue_veh = [] # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        self.travel_time_list = [] ### [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.start_node = None
        self.end_node = None

    def send_veh(self, t_now, a):
        self.queue_veh = [v for v in self.queue_veh if v!=a.id]
        self.ou_c = max(0, self.ou_c-1)
        if self.ltype=='real': self.travel_time_list.append((t_now, t_now-a.cl_enter_time))
    
    def receive_veh(self, agent_id):
        self.run_veh.append(agent_id)
        self.in_c = max(0, self.in_c-1)

    def run_link_model(self, t_now=None, agents=None):
        for agent_id in self.run_veh:
            if agents[agent_id].cl_enter_time < t_now - self.fft:
                self.queue_veh.append(agent_id)
        self.run_veh = [v for v in self.run_veh if v not in self.queue_veh]
        ### remaining spaces on link for the node model to move vehicles to this link
        self.st_c = self.store_cap - len(self.run_veh) - len(self.queue_veh) 
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600
    
    def update_travel_time(self, t_now, link_time_lookback_freq):
        self.travel_time_list = [(t_rec, dur) for (t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            new_weight = np.mean([dur for (_, dur) in self.travel_time_list])
            return self.start_nid+1, self.end_nid+1, new_weight
        else:
            return None, None, None