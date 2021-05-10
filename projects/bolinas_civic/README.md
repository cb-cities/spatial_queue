# Bolinas traffic simulation (CIVIC project)

### File structure
```
+-- environment.yml (Python packages from Anaconda)
+-- README.md
+-- run_traffic_simulation.py (generate simulation results: time-stamped link speed)
+-- query_path.py (query positions of player vehicles)
+-- network_inputs
|   +-- bolinas_edges_sim.csv (road network links geometry and properties)
|   +-- bolinas_nodes_sim.csv (road network nodes geometry and properties)
+-- demand_inputs
|   +-- od_csv (folder containing the OD files)
|   +-- simple_od.py (generate OD according to VPHH and visitor counts)
+-- simulation_outputs
|   +-- link_weights (folder containing time-stepped link weights)
```

### Installation
1. Create conda environment `conda env create -f environment.yml`
2. Install [`sp`](https://github.com/cb-cities/sp)
3. Make some directories `mkdir -p demand_inputs/od_csv simulation_outputs/link_weights simulation_outputs/log simulation_outputs/network`

### Usage
1. Run the simulation
    * `python -c "import run_traffic_simulation; run_traffic_simulation.run_traffic_simulation(vphh=1.5, visitor_cnts=300)"`
        * `vphh` refers to the numbers of vehicles per household for evacuation; `visitor_cnts` refers to the number of tourist vehicles.
    * It only needs to be run once per scenario.
2. Collect outputs (for use in other programs)
    * The time-stepped link weights can be found in the `simulation_outputs/link_weights` folder, where the key is the time step in seconds, and the value (list) is the link-level speed (in m/s, e.g., 11.8 m/s is about 25 mph).
        * The order of the link-level speed is the same as the order of links in `network_inputs/bolinas_edges_sim.csv` (both has 605 links).
3. Query for vehicle positions
    * `python -c "import query_path; query_path.query_path(vphh=1.5, visitor_cnts=300, player_origin=143, player_destin=193, start_time=100, end_time=900)"`
        * `vphh` and `visitor_cnts` are the same as the above
        * `player_origin` and `player_destin` refer to the start and end node (destination node) of the player, correspoinding to the road intersections `nid` field specified in `network_inputs/bolinas_nodes_sim.csv`.
        * `start_time` (in seconds) refers to the time that the player vehicle is at `player_origin`; `end_time` (in seconds) refers to the time that we want to know the vehicle positions, e.g., it could be 900 seconds (15 min) after the `start_time`.
    * This step can be used for querying player vehicle position for multiple times. For example, if want to know the position of a player starting from node 143 and ending at 193 at every 900s:
        * `python -c "import query_path; query_path.query_path(vphh=1.5, visitor_cnts=300, player_origin=143, player_destin=193, start_time=0, end_time=900)"`
            * It will show output "vehicle is on link 541 at 900 seconds. The end node ID of the current link is 122".
        * `python -c "import query_path; query_path.query_path(vphh=1.5, visitor_cnts=300, player_origin=122, player_destin=193, start_time=900, end_time=1800)"` Note the `player_origin` is set to the node where vehicle stops at 900s.
            * It will show output "vehicle is on link 543 at 1800 seconds. The end node ID of the current link is 20".
        * `python -c "import query_path; query_path.query_path(vphh=1.5, visitor_cnts=300, player_origin=20, player_destin=193, start_time=1800, end_time=2700)"`
            * It will show output "vehicle is on link 575 at 2700 seconds. The end node ID of the current link is 193".
4. Optional: Provide the alternative `write_path` paramter to `run_traffic_simulation.py` and `read_path` to `queue_path.py`. This will create/read from new the directory for the following files:
   
    ```
    +-- demand_inputs
    |   +-- od_csv
    |       +-- resident_visitor_od_vphh{vphh}_vistor{vistor_cnts}.csv
    +-- simulation_outputs
    |   +-- link_weights
    |       +-- link_speed_vphh{vphh}_vistor{vistor_cnts}.json
    |   +-- log
    |       +-- vphh{vphh}_vistor{vistor_cnts}.log
    |   +-- network
    |       +-- modified_network_edges_vphh{vphh}_vistor{vistor_cnts}.csv
    |       +-- network_links.json
    |       +-- node2link_dict.json
    ```
    `read_path` and `write_path` parameters are added to migrate the simulation code to AWS lambda, which has writing restriction for the file system.

    For example,
   
   * `python -c "import run_traffic_simulation; run_traffic_simulation.run_traffic_simulation(vphh=1.5, visitor_cnts=300, write_path='/tmp')"` will force the program to write all above files to `/tmp`.
        
    * `python -c "import query_path; query_path.query_path(vphh=1.5, visitor_cnts=300, player_origin=143, player_destin=193, start_time=0, end_time=900, read_path='/tmp')"` will read the required files from the directory `/tmp`.
