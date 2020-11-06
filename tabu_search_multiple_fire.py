import sys
import random
import numpy as np 
import pandas as pd 
import geopandas as gpd
import dta_meso_butte_tabu
from multiprocessing import Pool

rs = 0
random.seed(rs)
np.random.seed(rs)

def fitness(arg):

    dept_time_list, fs, fire_df = arg
    print(fs, dept_time_list)
    zone_dept_time = pd.DataFrame({'evac_zone': list(range(1, 15)), 'dept_time': dept_time_list})
    fitness_score = dta_meso_butte_tabu.main(dept_time_df=zone_dept_time, fs=fs, fire_df=fire_df)

    return (dept_time_list, fitness_score)

def getNeighborhood(bestCandidate):
    sNeighborhood = []
    for i in range(len(bestCandidate)):
        newCandidate = bestCandidate.copy()
        newCandidate[i] += 5*60
        # one_neighbor = bestCandidate.copy()
        # one_neighbor.loc[one_neighbor['evac_zone']==getattr(zone, 'evac_zone'), ['dept_time']] += 5*60
        sNeighborhood.append(newCandidate)
    return sNeighborhood

def fire_scenario_generator():
    fire_scenarios_df = pd.DataFrame([[0, 'init', -121.571127, 39.803276, 0, 4]], columns=['fire_scenario', 'start_zone', 'lon', 'lat', 'start_time', 'fire_speed'])
    
    alt_fire_scenarios_df = pd.DataFrame([
        ### scenario 1
        [1, 'Zone_2', 0, 4],
        [1, 'Zone_7', 0, 4],
        ### scenario 2
        [2, 'Zone_6', 0, 4],
        [2, 'Zone_8', 0, 4],
        [2, 'Zone_10', 0, 4],
        ### scenario 3
        [3, 'Zone_3', 0, 4],
        [3, 'Zone_5', 0, 4],
        [3, 'Zone_9', 0, 4]
    ], columns=['fire_scenario', 'start_zone', 'start_time', 'fire_speed'])
    
    evacuation_zone_gdf = gpd.read_file('projects/butte_osmnx/demand_inputs/digitized_evacuation_zone/digitized_evacuation_zone.shp')
    evacuation_zone_gdf['centroid'] = evacuation_zone_gdf.to_crs('epsg:3857')['geometry'].centroid
    evacuation_zone_gdf = evacuation_zone_gdf.set_geometry('centroid').to_crs('epsg:4326')
    evacuation_zone_gdf['c_x'] = evacuation_zone_gdf['centroid'].x
    evacuation_zone_gdf['c_y'] = evacuation_zone_gdf['centroid'].y
    alt_fire_scenarios_df = alt_fire_scenarios_df.merge(evacuation_zone_gdf[['zone_name', 'c_x', 'c_y']], how='left', left_on='start_zone', right_on='zone_name')
    alt_fire_scenarios_df = alt_fire_scenarios_df.rename(columns={'c_x': 'lon', 'c_y': 'lat'})
    fire_scenarios_df = pd.concat([fire_scenarios_df, alt_fire_scenarios_df[fire_scenarios_df.columns]])
    
    return fire_scenarios_df

def single_scenario_phase_search(fs=None, fire_df=None):

    ### initialization
    s0 = [0] * 14
    sBest = s0

    ### initial fit
    (_, sBest_fit) = fitness((sBest, fs, fire_df))
    # sys.exit(0)
    fit_df = pd.DataFrame( [['-'.join([str(x) for x in sBest]),'init', round(sBest_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])
    with open('tabu_search_r{}_fs{}.csv'.format(rs, fs), 'w') as tabu_search_outfile:
        tabu_search_outfile.write(",".join([str(x) for x in range(1, 15)])+",type," + "fitness" +"\n")
        tabu_search_outfile.write(",".join([str(x) for x in sBest])+",init,{:.2f}".format(sBest_fit) +"\n")
    
    step = 0
    # recursion_limit = 20
    while step < 50:
        step += 1
        sNeighborhood = getNeighborhood(sBest)
        sNeighborhood = [sCandidate for sCandidate in sNeighborhood if '-'.join([str(x) for x in sCandidate]) not in fit_df['zone_dept_time'].values.tolist()]
        # recursion_level = 0
        # if (len(sNeighborhood)==0) and (recursion_level<recursion_limit):
        if len(sNeighborhood)==0:
            second_best = fit_df.drop_duplicates(subset=['zone_dept_time'], keep=False)
            second_best = second_best[second_best['type']=='calc'].sort_values(by='fitness', ascending=True).iloc[0]
            fit_df = pd.concat([fit_df, pd.DataFrame(
                [[second_best['zone_dept_time'],'alt_step', second_best['fitness']]], columns= ['zone_dept_time', 'type', 'fitness'])])
            with open('tabu_search_r{}_fs{}.csv'.format(rs, fs), 'a') as tabu_search_outfile:
                tabu_search_outfile.write(second_best['zone_dept_time'].replace('-', ',')+",altstep," + "{}".format(second_best['fitness']) +"\n")
            # second_best = second_best[second_best['fitness']==np.max(second_best['fitness'])].iloc[0]['zone_dept_time'].split('-')
            second_best_dept_time = [int(x) for x in second_best['zone_dept_time'].split('-')]
            sNeighborhood = getNeighborhood(second_best_dept_time)
            sNeighborhood = [sCandidate for sCandidate in sNeighborhood if '-'.join([str(x) for x in sCandidate]) not in fit_df['zone_dept_time'].values.tolist()]
            # recursion_level += 1
        # else:
        #     'recursion limit reached and no better solution found'

        pool = Pool(np.min([6, len(sNeighborhood)]))
        res = pool.imap_unordered(fitness, [(sn, fs, fire_df) for sn in sNeighborhood])
        pool.close()
        pool.join()

        for (sCandidate, sCandidate_fit) in res:
            fit_df = pd.concat([fit_df, pd.DataFrame(
                [['-'.join([str(x) for x in sCandidate]),'calc', round(sCandidate_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])])
            with open('tabu_search_r{}_fs{}.csv'.format(rs, fs), 'a') as tabu_search_outfile:
                tabu_search_outfile.write(",".join([str(x) for x in sCandidate])+",calc," + "{:.2f}".format(sCandidate_fit) +"\n")
            if sCandidate_fit < sBest_fit:
                sBest = sCandidate
                sBest_fit = sCandidate_fit
            else:
                pass ### this neighbor is not better
        
        ### best step results
        fit_df = pd.concat([fit_df, pd.DataFrame(
            [['-'.join([str(x) for x in sBest]),'step', round(sBest_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])])
        with open('tabu_search_r{}_fs{}.csv'.format(rs, fs), 'a') as tabu_search_outfile:
            tabu_search_outfile.write(",".join([str(x) for x in sBest])+",step," + "{:.2f}".format(sBest_fit) +"\n")

def main():
    fire_scenarios_df = fire_scenario_generator()
    for fs in [0]:
        fire_df = fire_scenarios_df[fire_scenarios_df['fire_scenario'].isin([0, fs])].reset_index(drop=True)
        print(fs, fire_df)
        single_scenario_phase_search(fs=fs, fire_df=fire_df)

if __name__ == "__main__":
    main()
