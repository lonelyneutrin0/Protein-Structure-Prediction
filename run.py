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
    
    length = 21
    num_steps = 2749
    res = np.random.choice([0,1], (length))
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
    # if callback specified, results are already printed by callback
    if callback:
        return

    # if callback not specified, plot results
    plt.plot(run.param_inv_temps, run.optimal_energies)
    plt.axhline(run.optimalest_energy, color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\beta$')
    plt.ylabel('energy')
    plt.savefig('annealing.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    # plot num accepts and num rejects
    plt.plot(np.arange(num_steps) / 1e+3, run.param_num_rejects / 1e+3)
    plt.grid()
    plt.xlabel('metropolis step / $10^3$')
    plt.ylabel('number of acceptances / $10^3$')
    plt.savefig('num_accepts.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    optimal_conformation = run.optimalest_conformation
    if optimal_conformation.shape[1] != 3:
        raise ValueError("The optimal conformation must have 3 columns representing x, y, z coordinates.")
    
    # Plotting the optimal conformation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each residue as a point
    ax.scatter(optimal_conformation[:, 0], optimal_conformation[:, 1], optimal_conformation[:, 2], c='k', marker='o',s=100)
    
    # Optionally, connect the residues with lines to visualize the backbone
    for i in range(len(optimal_conformation) - 1):
        ax.plot([optimal_conformation[i, 0], optimal_conformation[i + 1, 0]],
                [optimal_conformation[i, 1], optimal_conformation[i + 1, 1]],
                [optimal_conformation[i, 2], optimal_conformation[i + 1, 2]], c='k', lw=1)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimal Protein Conformation')
    
    # Show the plot
    plt.savefig('conformation.png', dpi=800, bbox_inches='tight')
    plt.show()
    plt.close()



example(None)
