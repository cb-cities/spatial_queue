import os
import sys
import random
import numpy as np

### user
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demand_inputs.simple_od import generate_simple_od
from run_traffic_simulation import run_traffic_simulation
from query_path import query_path

### set random seed
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# generate_simple_od(vphh=1.5, visitor_cnts=0)
# run_traffic_simulation(vphh=1.5, visitor_cnts=0)
query_path(vphh=1.5, visitor_cnts=0, player_origin=143, player_destin=193, start_time=0, end_time=100)