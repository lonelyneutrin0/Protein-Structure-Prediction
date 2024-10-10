from src_np import GeneticAnnealer, artificial_protein
import numpy as np
import gzip 
import json 
testargs = { 
    'num_iterations': 100 , 
    'temp': 1,
    'num_annealers': 8, 
    'ml': 10000, 
    'quality_factor': 1.5,
    'lam': 3,
    'residues': artificial_protein(6),
    'index': 0
}
x = GeneticAnnealer(**testargs)
if __name__ == "__main__": 
    output = x.optimize()           
    print(output.optimal_energies[-1, -1])
    # with gzip.open("/ga/run.json.gz", "wt", encoding='utf-8') as f: 
    #     json.dump(output.to_dict(), f, indent=4)
    
