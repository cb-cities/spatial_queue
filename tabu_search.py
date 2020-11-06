import sys
import random
import numpy as np 
import pandas as pd 
import dta_meso_butte_tabu
from multiprocessing import Pool

rs = 0
random.seed(rs)
np.random.seed(rs)

def fitness(dept_time_list):

    zone_dept_time = pd.DataFrame({'evac_zone': list(range(1, 15)), 'dept_time': dept_time_list})
    fitness_score = dta_meso_butte_tabu.main(dept_time_df=zone_dept_time)

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

def main():

    ### initialization
    # s0 = pd.DataFrame([(-1, 0)] + [(k, 0) for k in range(1, 15)], columns=['evac_zone', 'dept_time'])
    s0 = [0] * 14
    sBest = s0
    # tabuList = []
    # tabuList.append(s0['dept_time'].values.tolist())

    ### initial fit
    (_, sBest_fit) = fitness(dept_time_list=sBest)
    fit_df = pd.DataFrame( [['-'.join([str(x) for x in sBest]),'init', round(sBest_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])
    with open('tabu_search_r{}.csv'.format(rs), 'w') as tabu_search_outfile:
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
            with open('tabu_search_r{}.csv'.format(rs), 'a') as tabu_search_outfile:
                tabu_search_outfile.write(second_best['zone_dept_time']+",altstep," + "{}".format(second_best['fitness']) +"\n")
            # second_best = second_best[second_best['fitness']==np.max(second_best['fitness'])].iloc[0]['zone_dept_time'].split('-')
            second_best_dept_time = [int(x) for x in second_best['zone_dept_time'].split('-')]
            sNeighborhood = getNeighborhood(second_best_dept_time)
            sNeighborhood = [sCandidate for sCandidate in sNeighborhood if '-'.join([str(x) for x in sCandidate]) not in fit_df['zone_dept_time'].values.tolist()]
            # recursion_level += 1
        # else:
        #     'recursion limit reached and no better solution found'

        pool = Pool(np.min([6, len(sNeighborhood)]))
        res = pool.imap_unordered(fitness, sNeighborhood)
        pool.close()
        pool.join()

        for (sCandidate, sCandidate_fit) in res:
            fit_df = pd.concat([fit_df, pd.DataFrame(
                [['-'.join([str(x) for x in sCandidate]),'calc', round(sCandidate_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])])
            with open('tabu_search_r{}.csv'.format(rs), 'a') as tabu_search_outfile:
                tabu_search_outfile.write(",".join([str(x) for x in sCandidate])+",calc," + "{:.2f}".format(sCandidate_fit) +"\n")
            if sCandidate_fit < sBest_fit:
                sBest = sCandidate
                sBest_fit = sCandidate_fit
            else:
                pass ### this neighbor is not better
        
        ### best step results
        fit_df = pd.concat([fit_df, pd.DataFrame(
            [['-'.join([str(x) for x in sBest]),'step', round(sBest_fit, 2)]], columns= ['zone_dept_time', 'type', 'fitness'])])
        with open('tabu_search_r{}.csv'.format(rs), 'a') as tabu_search_outfile:
            tabu_search_outfile.write(",".join([str(x) for x in sBest])+",step," + "{:.2f}".format(sBest_fit) +"\n")


if __name__ == "__main__":
    main()
