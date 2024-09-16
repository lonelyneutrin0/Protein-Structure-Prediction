#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src import annealer
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from src import get_energy, get_conformation


def example(callback):
    
    length = 13
    num_steps = 2749
    res = np.array([0,0,0,0,0,1,1,1,1,1])
    print(res.size)

    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'start_temp': 1.0,
        'end_temp': 10**(-12), 
        'gamma': 0.99,
        'lam': 3.0,
    }
    
    # perform run
    past = time.perf_counter()
    run = annealer(**kwargs)
    present = time.perf_counter()
    sec_elapsed = present - past
    print(f'Annealer run with {num_steps:.0f} steps took {sec_elapsed:.3f} seconds')
    print(f'Lowest Energy: {run.optimalest_energy} \n Optimal Energy: {run.optimal_energies[-1]}')
 
    # Run Data
    plt.plot(run.param_inv_temps, run.optimal_energies)
    plt.axhline(run.optimalest_energy, color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\frac{1}{T}$')
    plt.ylabel('energy')
    plt.savefig('run_data/annealing.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    # Acceptance vs Progress of Annealing
    plt.plot(np.arange(num_steps), run.param_num_rejects)
    plt.grid()
    plt.xlabel('metropolis step')
    plt.ylabel('number of acceptances')
    plt.savefig('run_data/num_accepts.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    # 2D Projection of Model
    optimal_conformation = run.optimalest_conformation
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
    ax.scatter(optimal_conformation[:, 0], optimal_conformation[:, 1], optimal_conformation[:, 2], c='k', marker='o',s=100)
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



example(None)
