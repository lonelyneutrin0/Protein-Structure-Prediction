#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src_np import n_annealer, artificial_protein
import numpy as np
import time
import os
import json
import gzip 

def example(input_protein):
    
    res = np.array([1 if char == 'A' else 0 for char in input_protein]) 
    
    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'start_temp': 1.0,
        'end_temp': 1e-12, 
        'gamma': 0.99,
        'lam': 3.0,
        'ml': 50000
    }
   
    num_steps = (int)(np.log10(kwargs['end_temp']/kwargs['start_temp'])/np.log10(kwargs['gamma']))
    # perform run
    past = time.perf_counter()
    run = n_annealer(**kwargs) 
    present = time.perf_counter()
    sec_elapsed = present - past
    # print(f'Time elapsed: {sec_elapsed}')
    return run
run_object = example(artificial_protein(7)).to_dict()
# example("ABAABBAAABAAAABABAAABAABBAABBBAABABBAABAAAAAAAAAABAAABA") #1FCA
with gzip.open(f"ssa/run.json.gz", "wt", encoding='utf-8') as f:
    
    print(f"Optimal Energy: {run_object['energies'][-1]}")
    json.dump(run_object, f, indent=4)