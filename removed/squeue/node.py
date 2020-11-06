from .link import Link

class Node:
    def __init__(self, nid=None, lon=None, lat=None, ntype=None, osmid=None, network=None):
        self.id = nid
        self.lon = lon
        self.lat = lat
        self.ntype = ntype
        self.osmid = osmid
        self.network = network
        ### derived
        # self.id_sp = self.id + 1
        self.in_links = {} ### {in_link_id: straight_ahead_out_link_id, ...}
        self.out_links = []
        self.go_vehs = [] ### veh that moves in this time step
        self.status = None

    def create_virtual_node(self):
        return Node(nid='vn{}'.format(self.id), lon=self.lon+0.001, lat=self.lat+0.001, ntype='v')

    def create_virtual_link(self):
        return Link('n{}_vl'.format(self.id), 100, 0, 0, 100000, 'v', 'vn{}'.format(self.id), self.id, 'LINESTRING({} {}, {} {})'.format(self.lon+0.001, self.lat+0.001, self.lon, self.lat))
    