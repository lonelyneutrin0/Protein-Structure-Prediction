#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src_np import n_annealer
from src_t import t_annealer
import matplotlib as mpl
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from src_np import artificial_protein


def example(s):
    
    fibonacci_index = 6
    res = artificial_protein(fibonacci_index)
    
    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'start_temp': 1.0,
        'end_temp': 1e-12, 
        'gamma': 0.99,
        'lam': 3.0,
        'ml': 100,
    }
   
    num_steps = (int)(np.log10(kwargs['end_temp']/kwargs['start_temp'])/np.log10(kwargs['gamma']))
    # perform run
    past = time.perf_counter()
    run = n_annealer(**kwargs) if s=='n' else t_annealer(**kwargs)
    present = time.perf_counter()
    sec_elapsed = present - past

    return run.optimal_energy

runs = np.zeros(10)
for i in range(runs.size): 
    runs[i] = example('n')

plt.scatter(np.arange(10), runs, marker='o', color='black')
plt.axhline(-29.474, color='red', linestyle=':')
plt.savefig('13.png', dpi=800, bbox_inches='tight')
print(f'Least Energy Found: {np.min(runs)}\n Average Energy: {np.average(runs)}\n Standard Deviation: {np.std(runs, ddof=1)}')