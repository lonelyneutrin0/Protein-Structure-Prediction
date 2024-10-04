import multiprocessing.pool
import torch 
from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing
import random
from rich import print




@dataclass 
class runOutput:
    annealers: list 
    energies: torch.Tensor
    conformations: torch.Tensor
    boltzmen: torch.Tensor
    alphas: torch.Tensor
    betas: torch.Tensor  
    run_accepts: torch.Tensor
    run_rejects: torch.Tensor
    
    """ 
    Run Output Container Class
    :param annealers: A list of Annealer objects used in the run
    :param energies: A tensor of energy values 
    :param conformations: A tensor of conformations
    :param ratios: A tensor of acceptance ratios 
    :param boltzmen: A tensor of boltzmann values 
    :param alpha: Bond angle vector 
    :param beta: Torsion angle vector
    :param run_accepts: Tensor of num_accepts per run
    :param run_rejects: Tensor of num_rejects per run
    """

# Optimizer Object
@dataclass
class solutionObject: 
    alpha: torch.tensor # N-2
    beta: torch.tensor # N-3
    optimal_energies: float
    optimal_conformations: torch.tensor
    temps: torch.tensor
    runOutputs: list[runOutput]
    
    
    """ 
    An object containing the solution data
    :param optimal_energies: Best energy value
    :param optimal_conformations: Best conformation
    :param temps: Temperatures 
    :param boltzmen: A tensor of boltzmann values 
    :param alpha: Bond angle vector 
    :param beta: Torsion angle vector
    :param num_accepts: The number of transitions
    :param num_rejects: The number of rejections
    """

# Annealer
@dataclass 
class Annealer: 
    temp: float
    lam: float
    ml: int
    index: int
    residues: torch.Tensor
    no_temps: int
    device_id: str=None
    alpha: torch.Tensor=None
    beta: torch.Tensor=None
    k_1: float = -1.0
    k_2: float = 0.5
    
    """
    :param temp: The temperature at which the annealing occurs
    :param lam: The tuned constant for neighbor generation 
    :param ml: Markov chain length 
    :param index: Iteration index
    :param residues: Amino acid residue sequence 
    :param no_temps: The number of iterations
    :param device_id: The core ID of the process 
    :param alpha: The assumed optimal alpha 
    :param beta: The assumed optimal beta
    :param k_1: Interaction strength parameter
    :param k_2: Interaction strength parameter 
    """

    def __eq__(self, other): 
        return torch.equal(self.alpha, other.alpha) and torch.equal(self.beta, other.beta)
    def get_coefficient(self) -> torch.tensor:
        """
        
        Return the coefficient determining the strength of interactions between two residues.
        If both are 1, then it returns 1.
        If one is 0, it returns 0.5.
        If both are 0, it returns 0.5.
        
        Output is in the form of a coefficient matrix. 
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
        
        # Produce the cosine vectors
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        
        # Initialize the positions matrix and set the first three positions.
        positions = torch.zeros((size, 3), dtype=torch.float64)
        positions[0] = torch.tensor([0, 0, 0], dtype=torch.float64)
        positions[1] = torch.tensor([0, 1, 0], dtype=torch.float64)
        positions[2] = positions[1] + torch.tensor([cos_alpha[0], sin_alpha[0], 0], dtype=torch.float64)
        
        
        #Compute the new coordinate values 
        dx = cos_alpha[1:size-1] * cos_beta[:size-2]
        dy = sin_alpha[1:size-1] * cos_beta[:size-2]
        dz = sin_beta[:size-2]
        
        # Iteratively update the positions using the recursive relation
        for i in range(3, size):
            positions[i] = positions[i-1]+torch.stack((dx,dy,dz), dim=1)[i-3]
        return positions
    
    def get_energy(self, alpha, beta) -> float: 
        """ 
        Protein Conformation Energy Function 
        
        :param alpha: Bond angle vector
        :param beta: Torsion angle vector
        
        The energy is split into backbone bending energy, torsion energy and energy based on the hydrophobic interactions of residues.
        The third energy is computed using matrices and the required mask. 
        """
        
        # Obtain the conformation of the protein
        conformation = self.get_conformation(alpha, beta)
        
        # Get the first two terms of the energy equation
        backbone_bending_energy = -self.k_1*torch.sum(torch.cos(alpha))
        torsion_energy = -self.k_2*torch.sum(torch.cos(beta))
        
        # Compute the pairwise norm differences to obtain a distance matrix, set diagonal elements to torch.inf to prevent div/0
        distance_matrix = torch.linalg.norm(conformation[:, None] - conformation, dim=-1)
        distance_matrix.fill_diagonal_(torch.inf)
        distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
        
        # Add all the terms together after computing the desired sum of the matrix
        total_energy = backbone_bending_energy + torsion_energy + torch.sum(torch.triu(4*distance_matrix*self.get_coefficient(), diagonal=2))
        return total_energy
    
    def anneal(self): 
        
        """
        :returns  output: annealerOutput
        :raises ValueError: if angle vectors are of improper dimensions
        """
        
        # Initializations
        if self.alpha is None: 
            self.alpha = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-2,))
        
        if self.beta is None: 
            self.beta = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-3,))
        
        if(self.alpha.shape[0] !=  self.residues.shape[0]-2 or self.beta.shape[0] != self.residues.shape[0]-3): 
            raise ValueError(f'The angle vectors are not of the appropriate dimensionality.{self.alpha.shape[0]}, {self.beta.shape[0]}')
        
        energies=torch.zeros(self.ml,)
        boltzmen = torch.zeros(self.ml,)
        conformations = torch.zeros((self.ml, self.residues.shape[0], 3))
        conformation = self.get_conformation(self.alpha, self.beta)
        rannums = torch.rand(self.ml,)
        accepts = torch.zeros(self.ml,)
        rejects = torch.zeros(self.ml,)
        energy = self.get_energy(self.alpha, self.beta)
        
        # Annealing Step 
        
        for i in range(self.ml): 
            # Log Current Values
            energies[i] = energy 
            conformations[i] = conformation
            
            #Generate neighbors
            random_i = torch.randint(low=0, high=self.alpha.shape[0]+self.beta.shape[0], size=(1,)).item()
            new_alpha_v, new_beta_v = torch.clone(self.alpha), torch.clone(self.beta)
            change = (torch.rand(1).item()-0.5)*torch.rand(1).item()*(1-self.index/self.no_temps)**self.lam
            if random_i >= self.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - self.alpha.shape[0]] = new_beta_v[random_i - self.alpha.shape[0]] + change if torch.abs(new_beta_v[random_i - self.alpha.shape[0]] + change) < torch.pi else new_beta_v[random_i - self.alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if torch.abs(new_alpha_v[random_i]+change) < torch.pi else new_alpha_v[random_i]-change
            
            # Generate neighbor conformations and energy values
            new_conformation = self.get_conformation(new_alpha_v, new_beta_v)
            new_energy = self.get_energy(new_alpha_v, new_beta_v)
            energy_change = new_energy-energy
            boltzmen[i] = torch.exp(-energy_change/self.temp)
            
            # Compare via the Boltzmann criterion
            if energy_change < 0 or boltzmen[i] > rannums[i]:
                 self.alpha = new_alpha_v
                 self.beta = new_beta_v
                 conformation = new_conformation
                 energy = new_energy
                 accepts[i] = 1
            else: 
                rejects[i] = 1
        num_accepts = torch.sum(accepts)
        num_rejects = torch.sum(rejects)
        
        # Return the output of the run in the form of an annealerOutput object
        returnargs = { 
            'annealer': self,
            'energies': energies, 
            'conformations': conformations,
            'boltzmen': boltzmen, 
            'alpha': self.alpha, 
            'beta': self.beta, 
            'num_accepts': num_accepts, 
            'num_rejects': num_rejects
        }
        return annealerOutput(**returnargs)


#  Return type for each markov chain step
@dataclass
class annealerOutput: 
    annealer: Annealer
    energies: torch.Tensor
    conformations: torch.Tensor
    boltzmen: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    num_accepts: float
    num_rejects: float
    
    
    """ 
    Annealer Output Container Class
    :param energies: A tensor of energy values 
    :param conformations: A tensor of conformations
    :param ratios: A tensor of acceptance ratios 
    :param boltzmen: A tensor of boltzmann values 
    :param alpha: Bond angle vector 
    :param beta: Torsion angle vector
    :param num_accepts: Number of transitions 
    :param num_rejects: Number of rejections
    """

def get_child(p1: Annealer, p2: Annealer)->Annealer:
    """ 
    :param p1: The first parent 
    :param p2: The second parent
    :returns daughter: A daughter annealer 
    """
    # Determine whether a mutation will take place
    mutation = True if torch.randint(1, 1001, (1,)).item() == 313 else False

    # Compute the daughter characteristics
    lam_d = (2*p1.lam*p2.lam)/(p1.lam + p2.lam) # Harmonic mean
    rano_a, rano_b = torch.randint(0, p1.alpha.shape[0]-1, (1,)).item(), torch.randint(0, p2.beta.shape[0]-1, (1,)).item()
    
    # Slice the daughters
    alpha_d =  torch.concatenate((p1.alpha[0:rano_a], p2.alpha[rano_a:])) 
    beta_d = torch.concatenate((p1.beta[0:rano_b], p2.beta[rano_b:]))   
    
    # Perturb the components of bond and torsion angle vectors in the case of a mutation
    if mutation: 
        change = (torch.rand(1).item()-0.5)*torch.rand(1).item()*(1-p1.index/p1.no_temps)**lam_d
        random_i = torch.randint(low=0, high=p1.alpha.shape[0]+p1.beta.shape[0], size=(1,)).item()
        if random_i >= p1.alpha.shape[0]:
            # Prevents out of bound errors
            beta_d[random_i - p1.alpha.shape[0]] = beta_d[random_i - p1.alpha.shape[0]] + change if torch.abs(beta_d[random_i - p1.alpha.shape[0]] + change) < torch.pi else beta_d[random_i - p1.alpha.shape[0]]-change
        else:
            # Prevents out of bound errors
            alpha_d[random_i] = alpha_d[random_i]+change if torch.abs(alpha_d[random_i]+change) < torch.pi else alpha_d[random_i]-change

    # Return the child as an Annealer object
    childargs = { 
        'temp': p1.temp,
        'lam': lam_d, 
        'ml': p1.ml,
        'residues': p1.residues,
        'no_temps': p1.no_temps,
        'index': p1.index,
        'alpha': alpha_d,
        'beta': beta_d,
        'device_id': p1.device_id
    }
    return Annealer(**childargs)

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
    index: int
    
    """
    :param num_iterations: Number of temperatures to run for 
    :param temp: The current temperature of the genetic annealer 
    :param num_annealers: The number of parallel annealers to run 
    :param ml: Markov chain length 
    :param quality_factor: Hyperparameter of the cooling schedule
    :param lam: Hyperparamter for neighborhood selection 
    :param residues: The amino acid sequence
    :param index: The current iteration index
    """
    
    def wrapper(self, cpu_id: int, temp: float, annealerObject: Annealer):
        """ 
        :param cpu_id: The ID of the core 
        :param temp: The current annealing temperature 
        :param annealerObject: A preinitialized annealer that is updated to the current temperature. None by default 
        :returns results: A list of annealerOutput objects
        """
        
        # If annealers are generated, update their core_id and temperature before annealing
        annealerObject.index = self.index
        annealerObject.temp = temp
        annealerObject.device_id = cpu_id
        return annealerObject.anneal()
    
    def temp_updater(self, current_temp: float, runData: runOutput):
        """
        Temperature Updater 
        It takes the standard deviation of energy and boltzmann acceptance ratios and uses it to adaptively calculate the next annealing cycle temperature 
        :param current_temp: T_i 
        :energy_std: Energy standard deviation of the T_i run 
        :acceptance mean: 
        """
        
        # Compute the std of energy and acceptance mean across the pool 
        energy_std = torch.std(runData.energies)
        acceptance_mean = torch.mean(runData.run_accepts).item()
        
        # Compute the change to temperature based on the Lam-Delosme cooling schedule
        diff = self.quality_factor*(1/energy_std)*(current_temp/energy_std)**2 * (4*acceptance_mean*(1-acceptance_mean)**2)/(2-acceptance_mean)**2
        
        # Return the new temperature 
        inv_t_new = (1/current_temp)+diff
        self.temp = 1/inv_t_new

    def crossover(self, runData: runOutput)->list: 
        """
        :param runData: The data of the run 
        :returns list<Annealer>: The function returns a list of daughter annealers which make up the new population. They are genetically similar to their parents with adequate variation. 
        """
        population = []
        
        # Compute the relative fitness of each parent
        J_i = runData.run_accepts/torch.sum(runData.run_accepts)
        fit = (J_i - torch.min(J_i))/(torch.max(J_i - torch.min(J_i))) # [0,1]
        # Sort in descending order of fitness
        cmpd_arr = sorted(list(zip(fit.tolist(), runData.annealers)), key=lambda x: x[0], reverse=True)
        fit, annealers = list(zip(*cmpd_arr))

        #Accept the parents with relative fitness greater than 0.5
        parents = []
        for i in range(len(fit)): 
            if fit[i] > 0.5: 
                parents.append(annealers[i])
        
        # Compute the weighted probability list based on the geometric progression
        probability_ratios = 2**torch.arange(0, len(parents), 1)
        probability_ratios = probability_ratios.tolist()[::-1]
        weighted_parents = [item for item, count in zip(parents,probability_ratios) for _ in range(count)]
        # Create the new population 
        while len(population) < len(annealers):
            p1 = weighted_parents[random.randint(0, len(weighted_parents)-1)] 
            p2 = weighted_parents[random.randint(0, len(weighted_parents)-1)]
            while (p1.__eq__(p2) and len(parents) != 1): 
                p2 = weighted_parents[random.randint(0, len(weighted_parents)-1)]
            child = get_child(p1,p2)
            energy_diff=child.get_energy(child.alpha, child.beta) - parents[-1].get_energy(parents[-1].alpha, parents[-1].beta)  
            if energy_diff < 0 or torch.exp(-energy_diff/self.temp) > torch.rand(1,).item(): 
                population.append(child)
        return population
          
    def run(self, annealers: list[Annealer]= None)->annealerOutput: 
        if annealers is None:
            annealers = [] 
            for i in range(self.num_annealers): 
                args = { 
                'temp': self.temp, 
                'lam': (self.lam-0.5) + torch.rand(1,).item(), # [lam-1, lam+1] 
                'ml': self.ml, 
                'residues': self.residues, 
                'index': self.index,
                'no_temps': self.num_iterations, 
                'device_id': i, 
                }
                annealers.append(Annealer(**args))
                
            
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
            run_accepts = torch.zeros(self.num_annealers)
            run_rejects = torch.zeros(self.num_annealers)
            annealers=[]
        
        # Collect run data
        for i in range(len(results)):
            annealers.append(results[i].annealer)
            run_energies[i] = results[i].energies
            run_conformations[i] = results[i].conformations
            run_boltzmen[i] = results[i].boltzmen
            run_optimal_alphas[i] = results[i].alpha
            run_optimal_betas[i] = results[i].beta
            run_accepts[i] = results[i].num_accepts
            run_rejects[i] = results[i].num_rejects
        
        return runOutput(annealers, run_energies, run_conformations,run_boltzmen, run_optimal_alphas, run_optimal_betas, run_accepts, run_rejects)
    
    def optimize(self)->solutionObject:
        outputs = []
        # Initializations
        temps = torch.zeros(self.num_iterations,)
        initial_run = self.run()
        outputs.append(initial_run)
        population_energies = torch.zeros(self.num_annealers ,self.ml*self.num_iterations)
        new_annealers = self.crossover(initial_run) 
        self.temp_updater(self.temp, initial_run)
        for i in range(self.num_iterations):
            temps[i]=(self.temp)
            run = self.run(new_annealers)
            self.index+=1
            outputs.append(run)
            population_energies[:, i*self.ml:(i+1)*self.ml] = run.energies
            new_annealers = self.crossover(run) 
            self.temp_updater(self.temp, run)
        solutionArgs = {  
            'alpha': run.alphas,
            'beta': run.betas,
            'optimal_energies': run.energies,
            'optimal_conformations': run.conformations,
            'temps': temps,
            'runOutputs': outputs
        }
        return solutionObject(**solutionArgs)
    

testargs = { 
    'num_iterations': 100, 
    'temp': 100, 
    'num_annealers': 8, 
    'ml': 10000, 
    'quality_factor': 1.5,
    'lam': 3,
    'residues': torch.Tensor([1,0,0,1,0,0,1,0,1,0,0,1,0]),
    'index': 0
}
x = GeneticAnnealer(**testargs)
if __name__ == "__main__": 
    output = x.optimize()           
    for i in range(len(output.runOutputs)-1): # Each run
        for j in range(testargs['num_annealers']):
            plt.title(label=f"Energy Values Across Annealers | Run: {i}")
            plt.plot(torch.arange(1, testargs['ml']+1, 1), output.runOutputs[i].energies[j], label=f'core_{j}')
        plt.legend() 
        plt.show()
        plt.close()
    