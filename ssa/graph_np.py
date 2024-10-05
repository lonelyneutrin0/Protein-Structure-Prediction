#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from src_np import n_annealer
import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess
from src_np import artificial_protein
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def example(input_protein):
    
    res = np.array([1 if char == 'A' else 0 for char in input_protein]) 
    
    # define annealing arguments, track time to run
    kwargs = {
        'residues': res, 
        'start_temp': 1.0,
        'end_temp': 1e-12, 
        'gamma': 0.99,
        'lam': 3.0,
        'ml': 100000,
    }
   
    num_steps = (int)(np.log10(kwargs['end_temp']/kwargs['start_temp'])/np.log10(kwargs['gamma']))
    # perform run
    past = time.perf_counter()
    run = n_annealer(**kwargs) 
    present = time.perf_counter()
    sec_elapsed = present - past
    print(f'Annealer run with {num_steps:.0f} steps took {sec_elapsed:.3f} seconds')
    print(f'\n Optimal Energy: {run.energies[-1]}')
    print(f'Alpha Vector: {run.alpha} \n Beta Vector: {run.beta}')
    
 
    # Run Data
    energies = run.energies.copy()
    energies[energies > energies[-1]+1] = energies[-1]+1
    plt.plot(run.p_inv_temps, energies)
    plt.axhline(run.optimal_energy, color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\frac{1}{T} $')
    plt.ylabel('energy')
    
    plt.savefig('run_data/annealing.png', dpi=800, bbox_inches='tight')
    
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
    pca = PCA(n_components=2)
    optimal_projection = pca.fit_transform(optimal_conformation)
    plt.scatter(optimal_projection[:, 0], optimal_projection[:, 1], c='k', marker='o')
    plt.plot(optimal_projection[:, 0], optimal_projection[:, 1], color='k', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("PCA Projection of Optimal Protein Structure")
    plt.savefig("run_data/projection.png", dpi=800, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Real Time Updating
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if optimal_conformation.shape[1] != 3:
        raise ValueError("The optimal conformation must have 3 columns representing x, y, z coordinates.")
    ax.cla()
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
    
    for i in range(run.conformations.shape[0]): 
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(run.conformations[i][:, 0], run.conformations[i][:, 1], run.conformations[i][:, 2], c='k', marker='o', s=100)
        ax.set_title(f'{i}th Conformation \n Energy: {run.energies[i]}')
        for j in range(run.conformations.shape[1]-1):
            ax.plot([run.conformations[i][j, 0], run.conformations[i][j + 1, 0]],
                    [run.conformations[i][j, 1], run.conformations[i][j + 1, 1]],
                    [run.conformations[i][j, 2], run.conformations[i][j + 1, 2]], c='k', lw=1)
        plt.savefig(f'frames/frame_{i:03d}.png')
        plt.close()
    
    ffmpeg_cmd = [
    'ffmpeg',
    '-r', '30', 
    '-i', f'frames/frame_%03d.png',  
    '-vcodec', 'libx264',  
    '-crf', '25', 
    '-pix_fmt', 'yuv420p', 
    'annealing.mp4'   
    ]
    
    subprocess.run(ffmpeg_cmd)
    return run.optimal_energy

example("ABAABBAAABAAAABABAAABAABBAABBBAABABBAABAAAAAAAAAABAAABA") #1FCA
# example("ABBABBABABBAB")
 