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
        'end_temp': 10**(-12), 
        'gamma': 0.99,
        'lam': 3.0,
        'ml': 10,
    }
   
    num_steps = (int)(np.log10(kwargs['end_temp']/kwargs['start_temp'])/np.log10(kwargs['gamma']))
    # perform run
    past = time.perf_counter()
    run = n_annealer(**kwargs) if s=='n' else t_annealer(**kwargs)
    present = time.perf_counter()
    sec_elapsed = present - past
    print(f'Annealer run with {num_steps:.0f} steps took {sec_elapsed:.3f} seconds')
    print(f'\n Optimal Energy: {run.energies[-1]}')
 
    # Run Data
    energies = run.energies
    energies[energies>(run.optimal_energy+10)] = 0
    plt.plot(run.p_inv_temps, run.energies)
    plt.axhline(run.optimal_energy, color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\frac{1}{T} $')
    plt.ylabel('energy')
    
    plt.savefig('run_data/annealing.png', dpi=800, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Acceptance vs Progress of Annealing
    plt.plot(np.arange(num_steps), run.p_num_accepts)
    plt.grid()
    plt.xlabel('metropolis step')
    plt.ylabel('number of acceptances')
    plt.savefig('run_data/num_accepts.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    # 2D Projection of Model
    optimal_conformation = run.optimal_conformation
    plt.figure(figsize=(8, 6))
    plt.scatter(optimal_conformation[:, 0], optimal_conformation[:, 1], color='black', marker='o', s=50)
    plt.plot(optimal_conformation[:, 0], optimal_conformation[:, 1], marker='o', color='black', linestyle='-')
    plt.title('2D Projection of the Protein Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig('run_data/projection.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    if optimal_conformation.shape[1] != 3:
        raise ValueError("The optimal conformation must have 3 columns representing x, y, z coordinates.")
    
    # Optimal Configuration
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(optimal_conformation.shape[0]):
        if i == 0: 
            ax.scatter(optimal_conformation[i, 0], optimal_conformation[i, 1], optimal_conformation[i, 2], c='y', marker='o',s=100)
            continue
        if(run.residues[i] == 1): 
            ax.scatter(optimal_conformation[i, 0], optimal_conformation[i, 1], optimal_conformation[i, 2], c='r', marker='o',s=100)
        else: 
            ax.scatter(optimal_conformation[i, 0], optimal_conformation[i, 1], optimal_conformation[i, 2], c='b', marker='o',s=100)
    for i in range(len(optimal_conformation) - 1):
        ax.plot([optimal_conformation[i, 0], optimal_conformation[i + 1, 0]],
                [optimal_conformation[i, 1], optimal_conformation[i + 1, 1]],
                [optimal_conformation[i, 2], optimal_conformation[i + 1, 2]], c='k', lw=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimal Protein Conformation')
    
    plt.savefig('run_data/conformation.png', dpi=800, bbox_inches='tight')
    plt.show()
    plt.close()
    return run.optimal_energy
    
example('n')