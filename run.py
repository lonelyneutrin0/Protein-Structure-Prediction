#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src import annealer
import matplotlib as mpl
import json 
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from src import get_energy, get_conformation, artificial_protein
f = open('params.json')
params = json.load(f)


def example():
    
    fibonacci_index = 6
    
    
    res = artificial_protein(fibonacci_index)
  
    
    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'start_temp': 1.0,
        'end_temp': 10**(-12), 
        'gamma': 0.99,
        'lam': 3.0,
        'ml': 50000,
    }
    if(len(params['alpha_vector']) == res.size-2): 
        kwargs['init_alpha'] = np.array(params['alpha_vector'])
    if(len(params['beta_vector']) == res.size-3): 
        kwargs['init_beta'] = np.array(params['beta_vector'])
    f.close()
    num_steps = (int)(np.log10(kwargs['end_temp']/kwargs['start_temp'])/np.log10(kwargs['gamma']))
    # perform run
    past = time.perf_counter()
    run = annealer(**kwargs)
    present = time.perf_counter()
    data = { 
        "alpha_vector": list(run.optimal_alpha), 
        "beta_vector": list(run.optimal_beta)
    }
    json_object = json.dumps(data, indent=2)
    with open("params.json", "w") as outfile: 
        outfile.write(json_object)
    return run.optimal_energies


