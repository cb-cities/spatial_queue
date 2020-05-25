import os
import logging
import multiprocessing as mp 

def my_func(x):
    return x, mp.current_process()._identity[0], os.environ['host']

def main(scale=1):
    pool = mp.Pool(3)
    res = pool.map(my_func, [i*scale for i in [1, 2, 3]])
    
    v, pid, nid = zip(*res)
    logger = logging.getLogger("test")
    logging.basicConfig(filename='{}.log'.format(scale), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(v)
    logging.info(pid)
    logging.info(nid)

if __name__ == "__main__":
    main()