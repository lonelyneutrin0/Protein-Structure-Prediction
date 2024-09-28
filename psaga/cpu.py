import multiprocessing.pool
import torch 
from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing
import random
# Defining required classes and methods 

# Optimizer Object
@dataclass
class solutionObject: 
    alpha: torch.tensor # N-2
    beta: torch.tensor # N-3
    optimal_energies: float
    optimal_conformations: torch.tensor
    temps: torch.tensor


    """ 
    :returns optimal_energies: Best energy value
    :returns optimal_conformations: Best conformation
    :returns temps: Temperatures 
    :returns boltzmen: A tensor of boltzmann values 
    :returns alpha: Bond angle vector 
    :returns beta: Torsion angle vector
    """

#  Return type for each markov chain step
@dataclass
class annealerOutput: 
    energies: torch.Tensor
    conformations: torch.Tensor
    boltzmen: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor

    
    """ 
    Annealer Output Container Class
    :returns energies: A tensor of energy values 
    :returns conformations: A tensor of conformations
    :returns ratios: A tensor of acceptance ratios 
    :returns boltzmen: A tensor of boltzmann values 
    :returns alpha: Bond angle vector 
    :returns beta: Torsion angle vector
    """

@dataclass 
class runOutput: 
    energies: torch.Tensor
    conformations: torch.Tensor
    boltzmen: torch.Tensor
    alphas: torch.Tensor
    betas: torch.Tensor  
    
    """ 
    Run Output Container Class
    :returns energies: A tensor of energy values 
    :returns conformations: A tensor of conformations
    :returns ratios: A tensor of acceptance ratios 
    :returns boltzmen: A tensor of boltzmann values 
    :returns alpha: Bond angle vector 
    :returns beta: Torsion angle vector
    """

# Annealer
@dataclass 
class Annealer: 
    temp: float
    lam: float
    ml: int
    residues: torch.Tensor
    no_temps: int
    device_id: str=None
    alpha: torch.Tensor=None
    beta: torch.Tensor=None
    k_1: float = -1.0
    k_2: float = 0.5

    
    
    def get_coefficient(self) -> torch.tensor:
        """
        
        Return the coefficient determining the strength of interactions between two residues
        If both are 1, then it returns 1 
        If one is 0, it returns 0.5 
        If both are 0, it returns 0.5
        
        :param i: First residue 
        :param j: Second residue 
        
        """
        coeff_matrix = self.residues[:, None] * self.residues 
        coeff_matrix[coeff_matrix == 0] = 0.5
        return coeff_matrix
    
    def get_conformation(self, alpha, beta) -> torch.tensor: 
        
        """
        
        It returns a vector of residue positions based on the bond and torsion angles
        
        :param alphas: Vector of bond angles 
        :param betas: Vector of torsion angles 
        :raises ValueError: If the size is less than 3
        
        """
        size=self.residues.shape[0]
        if size< 3: 
            raise ValueError(f'Enter a size greater than 3 to proceed')
        
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        positions = torch.zeros((size, 3), dtype=torch.float64)
        positions[0] = torch.tensor([0, 0, 0], dtype=torch.float64)
        positions[1] = torch.tensor([0, 1, 0], dtype=torch.float64)
        positions[2] = positions[1] + torch.tensor([cos_alpha[0], sin_alpha[0], 0], dtype=torch.float64)
        
        
        #compute the new coordinate values 
        dx = cos_alpha[1:size-1] * cos_beta[:size-2]
        dy = sin_alpha[1:size-1] * cos_beta[:size-2]
        dz = sin_beta[:size-2]
        for i in range(3, size):
            positions[i] = positions[i-1]+torch.stack((dx,dy,dz), dim=1)[i-3]
        return positions
    
    def get_energy(self, alpha, beta) -> float: 
        """ 
        Protein Conformation Energy Function 
        
        :param alpha: Bond angle vector
        :param beta: Torsion angle vector
        :param residues: Array of the residue variables 
        :param conformation: A matrix of residue positions
        :param coeff: A matrix of coefficients based on the nature of the sequence
        :param k_1: Weight parameter for backbone bending energy 
        :param k_2: Weight parameter for the torsion energy
        :returns: float containing the potential energy value
        
        The energy is split into backbone bending energy, torsion energy and energy based on the hydrophobic interactions of residues.
        The third energy is computed using matrices and the required mask. 
        """
        
        conformation = self.get_conformation(alpha, beta)
        backbone_bending_energy = -self.k_1*torch.sum(torch.cos(alpha))
        torsion_energy = -self.k_2*torch.sum(torch.cos(beta))
        distance_matrix = torch.linalg.norm(conformation[:, None] - conformation, dim=-1)
        distance_matrix.fill_diagonal_(torch.inf)
        distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
        total_energy = backbone_bending_energy + torsion_energy + torch.sum(torch.triu(4*distance_matrix*self.get_coefficient(), diagonal=2))
        return total_energy
    
    def anneal(self) -> annealerOutput: 
        
        """
        :param temp: The temperature of the annealing step 
        :param gamma: Hyperparamter for neighborhood generation 
        :param ml: Markov chain length 
        :params k_1, k_2: Energy hyperparameters  
        :returns: annealerOutput
        :raises ValueError: if angle vectors are of improper dimensions
        """
        
        # Initializations
        if self.alpha is None: 
            self.alpha = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-2,))
        
        if self.beta is None: 
            self.beta = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-3,))
        
        if(self.alpha.shape[0] !=  self.residues.shape[0]-2 or self.beta.shape[0] != self.residues.shape[0]-3): 
            raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
        
        energies=torch.zeros(self.ml,)
        boltzmen = torch.zeros(self.ml,)
        conformations = torch.zeros((self.ml, self.residues.shape[0], 3))
        conformation = self.get_conformation(self.alpha, self.beta)
        rannums = torch.rand(self.ml,)
        
        energy = self.get_energy(self.alpha, self.beta)
        
        # Annealing Step 
        
        for i in range(self.ml): 
            # Log Current Values
            energies[i] = energy 
            conformations[i] = conformation
            
            #Update to neighbors 
            random_i = torch.randint(low=0, high=self.alpha.shape[0]+self.beta.shape[0], size=(1,)).item()
            new_alpha_v, new_beta_v = torch.clone(self.alpha), torch.clone(self.beta)
            change = (torch.rand(1).item()-0.5)*torch.rand(1).item()*(1-i/self.no_temps)**self.lam
            if random_i >= self.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - self.alpha.shape[0]] = new_beta_v[random_i - self.alpha.shape[0]] + change if torch.abs(new_beta_v[random_i - self.alpha.shape[0]] + change) < torch.pi else new_beta_v[random_i - self.alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if torch.abs(new_alpha_v[random_i]+change) < torch.pi else new_alpha_v[random_i]-change
            
            new_conformation = self.get_conformation(new_alpha_v, new_beta_v)
            new_energy = self.get_energy(new_alpha_v, new_beta_v)
            energy_change = new_energy-energy
            boltzmen[i] = torch.exp(-torch.clip(energy_change/self.temp, -10, 10))
            
            if energy_change < 0 or boltzmen[i] > rannums[i]:
                 self.alpha = new_alpha_v
                 self.beta = new_beta_v
                 conformation = new_conformation
                 energy = new_energy
        returnargs = { 
            'energies': energies, 
            'conformations': conformations,
            'boltzmen': boltzmen, 
            'alpha': self.alpha, 
            'beta': self.beta, 
        }
        return annealerOutput(**returnargs)

# Genetic Algorithm Handler 
@dataclass
class GeneticAnnealer: 
    num_iterations: int 
    temp: float
    num_annealers: int 
    ml: int 
    quality_factor: float
    lam: float
    residues: torch.Tensor
    
    """
    :param num_iterations: Number of temperatures to run for 
    :param temp: The current temperature of the genetic annealer 
    :param num_annealers: The number of parallel annealers to run 
    :param ml: Markov chain length 
    :param quality_factor: Hyperparameter of the cooling schedule
    :param lam: Hyperparamter for neighborhood selection 
    :param residues: The amino acid sequence
    """

    def wrapper(self, cpu_id: int, temp: float, annealerObject: Annealer=None):
        """
        Wrapper function for mapping multiprocessing 
        :param cpu_id: The ID of the core 
        :param temp: The current annealing temperature 
        :param annealerObject: A preinitialized annealer that is updated to the current temperature. None by default 
        """
        if annealerObject is None:  
            args = { 
                'temp': temp, 
                'lam': self.lam, 
                'ml': self.ml, 
                'residues': self.residues, 
                'no_temps': self.num_iterations, 
                'device_id': cpu_id, 
            }
            return Annealer(**args).anneal()
        annealerObject.temp = temp
        annealerObject.device_id = cpu_id
        return annealerObject.anneal()
    
    def temp_updater(self, current_temp: float, runData: runOutput)->float:
        """
        Temperature Updater 
        It takes the standard deviation of energy and boltzmann acceptance ratios and uses it to adaptively calculate the next annealing cycle temperature 
        :param current_temp: T_i 
        :energy_std: Energy standard deviation of the T_i run 
        :acceptance mean: 
        """
        energy_std = torch.std(runData.energies)
        acceptance_mean = torch.mean(runData.boltzmen)
        inv_t_new = (1/current_temp)+self.quality_factor*(1/energy_std)*(current_temp/energy_std)**2 * (4*acceptance_mean*(1-acceptance_mean)**2)/(2-acceptance_mean)**2 
        self.temp = 1/inv_t_new
    
    def selective_breeder(
            self, 
            runData: runOutput
    
    )->list: 
        """ 
        Culls unfit annealers and duplicates fit ones
        :param runData: runOutput object with information about the run and annealers 
        """
        annealers = []
      
        mu_boltzmann = torch.mean(runData.boltzmen, dim=1) # Computes the mean acceptance ratio across each annealing run 
        J_p = mu_boltzmann/torch.sum(runData.boltzmen) 
        
        fitness_matrix = (J_p - torch.min(J_p))/(torch.max(J_p - torch.min(J_p))) # Computes the relative fitness of each annealing run, ml x 1 matrix
        print(f' Fitness: {fitness_matrix}')
        print(f'Energies: {runData.energies[:, -1]}')
        # Choose the fit annealers, cull the rest
        for i in range(fitness_matrix.shape[0]): 
            if(fitness_matrix[i] > 0.5):  
                objArgs = { 
                    'temp': self.temp,
                    'lam': self.lam,
                    'ml': self.ml,
                    'residues': self.residues, 
                    'no_temps': self.num_iterations,
                    'alpha': runData.alphas[i],
                    'beta': runData.betas[i]
                }
                annealers.append(Annealer(**objArgs))
        
        # Breeding new solutions 
        no_offspring = self.num_annealers-len(annealers)
        for i in range(no_offspring): 
            
            # Select a random fit parent and change the alpha/beta vectors a bit to produce a child
            parent = annealers[random.randint(0,len(annealers)-1)]
            random_i = torch.randint(low=0, high=parent.alpha.shape[0]+parent.beta.shape[0], size=(1,)).item()
            new_alpha_v, new_beta_v = torch.clone(parent.alpha), torch.clone(parent.beta)
            change = (torch.rand(1).item()-0.5)*torch.rand(1).item()*(1-i/self.num_iterations)**self.lam
            if random_i >= parent.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - parent.alpha.shape[0]] = new_beta_v[random_i - parent.alpha.shape[0]] + change if torch.abs(new_beta_v[random_i - parent.alpha.shape[0]] + change) < torch.pi else new_beta_v[random_i - parent.alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if torch.abs(new_alpha_v[random_i]+change) < torch.pi else new_alpha_v[random_i]-change
            
            childargs = { 
                'temp': self.temp,
                'lam': self.lam,
                'ml': self.ml,
                'residues': self.residues, 
                'no_temps': self.num_iterations,
                'alpha': new_alpha_v,
                'beta': new_beta_v
            }
            child = Annealer(**childargs)
            annealers.append(child)
        
        return annealers
          
    def run(self, annealers: list=None)->annealerOutput: 
        # Checks to make sure the number of cores is sufficient
        if self.num_annealers > multiprocessing.cpu_count(): 
            raise ValueError(f'That value exceeds the number of available cores.')
        
        # Create the desired number of annealers and runs the first step of annealing 
        
        cpu_ids=list(range(self.num_annealers))
        
        with multiprocessing.Pool(processes=self.num_annealers) as pool: 
            if annealers is None:
                results = pool.starmap(self.wrapper, zip(cpu_ids, [self.temp] * self.num_annealers))
            else:
                results = pool.starmap(self.wrapper, zip(cpu_ids, [self.temp] * self.num_annealers, annealers))
            # Results is an array of annealerOutput objects
            run_energies = torch.zeros(self.num_annealers, self.ml) 
            run_conformations = torch.zeros(self.num_annealers, self.ml, self.residues.shape[0], 3) 
            run_boltzmen = torch.zeros(self.num_annealers, self.ml) 
            run_optimal_alphas = torch.zeros(self.num_annealers, self.residues.shape[0]-2) 
            run_optimal_betas = torch.zeros(self.num_annealers, self.residues.shape[0]-3) 
        
        for i in range(len(results)): 
            run_energies[i] = results[i].energies
            run_conformations[i] = results[i].conformations
            run_boltzmen[i] = results[i].boltzmen
            run_optimal_alphas[i] = results[i].alpha
            run_optimal_betas[i] = results[i].beta
        return runOutput(run_energies, run_conformations,run_boltzmen, run_optimal_alphas, run_optimal_betas)
    
    def optimize(self)->solutionObject:
        # Initializations
        temps = torch.zeros(self.num_iterations,)
        initial_run = self.run()
        
        new_annealers = self.selective_breeder(initial_run) 
        self.temp_updater(self.temp, initial_run)
        for i in range(self.num_iterations): 
            temps[i]=(self.temp)
            run = self.run(new_annealers)
            new_annealers = self.selective_breeder(run) 
            self.temp_updater(self.temp, run)
            
        solutionArgs = { 
            'alpha': run.alphas,
            'beta': run.betas,
            'optimal_energies': run.energies,
            'optimal_conformations': run.conformations,
            'temps': temps
        }
        return solutionObject(**solutionArgs)
    

testargs = { 
    'num_iterations': 50, 
    'temp': 1, 
    'num_annealers': 5, 
    'ml': 1000, 
    'quality_factor': 1.5,
    'lam': 3,
    'residues': torch.Tensor([1,0,0,1,0,0,1,0,1,0,0,1,0])
}
x = GeneticAnnealer(**testargs)
if __name__ == "__main__": 
    output = x.optimize()
    print(output.temps)
    for i in range(x.num_annealers): 
       plt.plot(torch.arange(1, x.ml+1, 1), output.optimal_energies[i, :])
    plt.show()