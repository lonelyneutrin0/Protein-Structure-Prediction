#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src import annealer
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time


def example(callback):
    
    # define matrix Q, length x length with random elements between -1/2 and 1/2
    length = 10
    num_steps = 200_000
    res = np.random.choice([0,1], (length))
    
    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'num_iterations': num_steps, 
        'high_temp': 10.0,  
        'low_temp': 1.0, 
    }
    
    # perform run
    past = time.perf_counter()
    run = annealer(**kwargs)
    present = time.perf_counter()
    sec_elapsed = present - past
    print(f'Annealer run with {num_steps:.0f} steps took {sec_elapsed:.3f} seconds')
    
    # if callback specified, results are already printed by callback
    if callback:
        return

    # if callback not specified, plot results
    plt.plot(run.param_inv_temps, run.param_energies)
    plt.axhline(run.param_o_min, color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\beta$')
    plt.ylabel('energy')
    plt.savefig('annealing.png', dpi=800, bbox_inches='tight')
    plt.close()

    # plot num accepts and num rejects
    plt.plot(np.arange(num_steps) / 1e+3, run.param_num_accepts / 1e+3)
    plt.grid()
    plt.xlabel('metropolis step / $10^3$')
    plt.ylabel('number of acceptances / $10^3$')
    plt.savefig('num_accepts.png', dpi=800, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    mpl.use('Agg')

    # run example without a callback
    example(None)
