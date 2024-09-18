from src import annealer, artificial_protein
import time 
import numpy as np
fibonacci_index = 7

num_steps = 2749
res = artificial_protein(fibonacci_index)


# define annealing arguments, track time to run
kwargs = {
    'residues': res, 
    'start_temp': 1.0,
    'end_temp': 10**(-12), 
    'gamma': 0.999,
    'lam': 3.0,
}
i = 0 
results = np.zeros(100)
while i < 100: 
    run = annealer(**kwargs)    
    results[i] = run.optimalest_energy
    i+=1
print(np.min(results))
