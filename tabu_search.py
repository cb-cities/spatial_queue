import sys
import random
import numpy as np 
import pandas as pd 
import dta_meso_butte_tabu
from multiprocessing import Pool

rs = 0
random.seed(rs)
np.random.seed(rs)

def fitness(zone_dept_time):

    fitness = dta_meso_butte_tabu.main(dept_time_df=zone_dept_time)

    return (zone_dept_time, fitness)

def getNeighborhood(bestCandidate):
    sNeighborhood = []
    for zone in bestCandidate.itertuples():
        one_neighbor = bestCandidate.copy()
        one_neighbor.loc[one_neighbor['evac_zone']==getattr(zone, 'evac_zone'), ['dept_time']] += 5*60
        sNeighborhood.append(one_neighbor)
    return sNeighborhood

def main():

    ### initialization
    s0 = pd.DataFrame([(-1, 0)] + [(k, 0) for k in range(1, 15)], columns=['evac_zone', 'dept_time'])
    sBest = s0
    tabuList = []
    tabuList.append(s0['dept_time'].values.tolist())

    ### initial fit
    sBest_fit = dta_meso_butte_tabu.main(dept_time_df=sBest)
    with open('tabu_search_r{}.csv'.format(rs), 'w') as tabu_search_outfile:
        tabu_search_outfile.write("-1," + ",".join([str(x) for x in range(1, 15)])+",type," + "fitness" +"\n")
        tabu_search_outfile.write(",".join([str(x) for x in sBest['dept_time'].values.tolist()])+",init," + "{:.2f}".format(sBest_fit) +"\n")
    
    step = 0
    while step < 5:
        step += 1
        sNeighborhood = getNeighborhood(sBest)
        sNeighborhood = [sCandidate for sCandidate in sNeighborhood if sCandidate['dept_time'].values.tolist() not in tabuList]

        pool = Pool(np.min([18, len(sNeighborhood)]))
        res = pool.imap_unordered(fitness, sNeighborhood)
        pool.close()
        pool.join()

        for (sCandidate, sCandidate_fit) in res:
            with open('tabu_search_r{}.csv'.format(rs), 'a') as tabu_search_outfile:
                tabu_search_outfile.write(",".join([str(x) for x in sCandidate['dept_time'].values.tolist()])+",calc," + "{:.2f}".format(fitness) +"\n")
            if sCandidate_fit > sBest_fit:
                sBest = sCandidate
                sBest_fit = sCandidate_fit
            else:
                pass ### this neighbor is not better
        
        ### best step results
        with open('tabu_search_r{}.csv'.format(rs), 'a') as tabu_search_outfile:
            tabu_search_outfile.write(",".join([str(x) for x in sBest['dept_time'].values.tolist()])+",step," + "{:.2f}".format(sBest_fit) +"\n")


if __name__ == "__main__":
    main()
