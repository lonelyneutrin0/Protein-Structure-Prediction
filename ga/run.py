from ga.src_t import GeneticAnnealer
import torch 
import gzip 
import json 
testargs = { 
    'num_iterations': 10, 
    'temp': 1000,
    'num_annealers': 8, 
    'ml': 1000, 
    'quality_factor': 1.5,
    'lam': 3,
    'residues': torch.tensor([1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]),
    'index': 0
}
x = GeneticAnnealer(**testargs)
if __name__ == "__main__": 
    output = x.optimize()           
    print(output.optimal_energies[-1, -1])
    with gzip.open("/ga/run.json.gz", "wt", encoding='utf-8') as f: 
        json.dump(output.to_dict(), f, indent=4)
    
