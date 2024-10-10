import multiprocessing.pool
import numpy as np 
from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing
import random
from rich import print
import os
import json 
import gzip 
from numpy.typing import ArrayLike

class runOutput:
    annealers: list 
    energies: ArrayLike
    conformations: ArrayLike
    boltzmen: ArrayLike
    alphas: ArrayLike
    betas: ArrayLike  
    run_accepts: ArrayLike
    run_rejects: ArrayLike
    
    """ 
    Run Output Container Class
    :param annealers: A list of Annealer objects used in the run
    :param energies: A tensor of energy values 
    :param conformations: A tensor of conformations
    :param boltzmen: A tensor of boltzmann values 
    :param alphas: Bond angle vector 
    :param betas: Torsion angle vector
    :param run_accepts: Tensor of num_accepts per run
    :param run_rejects: Tensor of num_rejects per run
    """
    def __init__(self, annealers:list, energies:ArrayLike, conformations:ArrayLike, boltzmen:ArrayLike, alphas:ArrayLike, betas:ArrayLike, run_accepts:ArrayLike, run_rejects:ArrayLike): 
        self.annealers = annealers
        self.energies = energies
        self.conformations = conformations
        self.boltzmen = boltzmen
        self.alphas = alphas
        self.betas = betas
        self.run_accepts = run_accepts
        self.run_rejects = run_rejects
    
    # Annealers are not included
    def to_dict(self): 
        return { 
            'energies': self.energies.tolist(), 
            'conformations': self.conformations.tolist(),
            'boltzmen': self.boltzmen.tolist(), 
            'alphas': self.alphas.tolist(), 
            'betas': self.betas.tolist(),
            'run_accepts': self.run_accepts.tolist(), 
            'run_rejects': self.run_rejects.tolist()
        }

# Optimizer Object
class solutionObject: 
    alpha: ArrayLike # N-2
    beta: ArrayLike # N-3
    optimal_energies: float
    optimal_conformations: ArrayLike
    temps: ArrayLike
    runOutputs: list
    
    
    """ 
    An object containing the solution data
    :param optimal_energies: Best energy value
    :param optimal_conformations: Best conformation
    :param temps: Temperatures  
    :param alpha: Bond angle vector 
    :param beta: Torsion angle vector
    :param runOutput: A list of runOutput objects
    """
    
    
    def __init__(self, alpha:ArrayLike, beta:ArrayLike, optimal_energies:ArrayLike, optimal_conformations:ArrayLike, temps:ArrayLike, runOutputs:list):
        self.alpha = alpha
        self.beta = beta
        self.optimal_energies = optimal_energies
        self.optimal_conformations = optimal_conformations
        self.temps = temps
        self.runOutputs = runOutputs
    
    # A function for JSON serialization
    def to_dict(self): 
        return { 
            'alpha': self.alpha.tolist(), 
            'beta': self.beta.tolist(), 
            'optimal_energies': self.optimal_energies.tolist(), 
            'optimal_conformations': self.optimal_conformations.tolist(),
            'temps': self.temps.tolist(), 
            'runOutputs': self.runOutputs
        }

@dataclass 
class Annealer: 
    temp: float
    lam: float
    ml: int
    index: int
    residues: ArrayLike
    no_temps: int
    device_id: str=None
    alpha: ArrayLike=None
    beta: ArrayLike=None
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
    def __init__(self, temp:float, lam:float, ml:int, index:int, residues:ArrayLike, no_temps:int, device_id=None, alpha:ArrayLike=None, beta:ArrayLike=None, k_1:float=-1.0, k_2:float=0.5):
        self.temp = temp
        self.lam = lam 
        self.ml = ml
        self.index = index
        self.residues = residues
        self.no_temps = no_temps
        self.device_id = "" if device_id is None else device_id
        self.alpha = -np.pi + np.random.uniform(size=(residues.shape[0]-2,))*2*np.pi if alpha is None else alpha
        self.beta = -np.pi + np.random.uniform(size=(residues.shape[0]-3,))*2*np.pi if beta is None else beta 
        self.k_1 = k_1 
        self.k_2 = k_2 
    
    def __eq__(self, other): 
        return (self.alpha == other.alpha).all() and (self.beta == other.beta).all() and self.lam == other.lam
    
    def get_coefficient(self) -> ArrayLike:
        """
        Return the coefficient determining the strength of interactions between two residues.
        If both are 1, then it returns 1.
        If one is 0, it returns 0.5.
        If both are 0, it returns 0.5.
        
        Output is in the form of a coefficient matrix. 
        """
        residues_ = self.residues.astype(float)
        coeff_matrix = residues_[:, None] * residues_ 
        coeff_matrix[coeff_matrix == 0] = 0.5
        return coeff_matrix
    
    def get_conformation(self, alpha, beta) -> ArrayLike: 
        
        """
        
        It returns a vector of residue positions based on the bond and torsion angles
        
        :param alphas: Vector of bond angles 
        :param betas: Vector of torsion angles 
        :raises ValueError: If the size is less than 3
        
        """
        
        size=self.residues.shape[0]
        
        # Ensure the residue is of appropriate size 
        if size< 3: 
            raise ValueError(f'Enter a size greater than 3 to proceed')
        
        # Produce the cosine vectors
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        sin_alpha = np.sin(alpha)
        sin_beta = np.sin(beta)
        
        # Initialize the positions matrix and set the first three positions.
        positions = np.zeros((size, 3), dtype=np.float64)
        positions[0] = np.array([0, 0, 0])
        positions[1] = np.array([0, 1, 0])
        positions[2] = positions[1] + np.array([cos_alpha[0], sin_alpha[0], 0])
        
        # Iteratively update the positions using the recursive relation
        for i in range(size-3): 
            positions[i+3] = positions[i+2]+np.array([cos_alpha[i+1]*cos_beta[i], sin_alpha[i+1]*cos_beta[i], sin_beta[i]]) 
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
        backbone_bending_energy = -self.k_1*np.sum(np.cos(alpha))
        torsion_energy = -self.k_2*np.sum(np.cos(beta))
        
        # Compute the pairwise norm differences to obtain a distance matrix, set diagonal elements to np.inf to prevent div/0
        distance_matrix = np.linalg.norm(conformation[:, None] - conformation, axis=-1)
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
        
        # Add all the terms together after computing the desired sum of the matrix
        total_energy = backbone_bending_energy + torsion_energy + np.sum(np.triu(4*distance_matrix*self.get_coefficient(), k=2))
        return total_energy
    
    def anneal(self): 
        
        """
        :returns  output: annealerOutput
        :raises ValueError: if angle vectors are of improper dimensions
        """
        if(self.alpha.shape[0] !=  self.residues.shape[0]-2 or self.beta.shape[0] != self.residues.shape[0]-3): 
            raise ValueError(f'The angle vectors are not of the appropriate dimensionality.{self.alpha.shape[0]}, {self.beta.shape[0]}')
        
        energies=np.zeros(self.ml,)
        boltzmen = np.zeros(self.ml,)
        conformations = np.zeros((self.ml, self.residues.shape[0], 3))
        conformation = self.get_conformation(self.alpha, self.beta)
        rannums = np.random.uniform(size=(self.ml,))
        accepts = np.zeros(self.ml,)
        rejects = np.zeros(self.ml,)
        energy = self.get_energy(self.alpha, self.beta)

        # Annealing Step 
        
        for i in range(self.ml): 
            # Log Current Values
            energies[i] = energy 
            conformations[i] = conformation
            
            #Generate neighbors
            random_i = np.random.randint(low=0, high=self.alpha.shape[0]+self.beta.shape[0], size=(1,)).item()
            new_alpha_v, new_beta_v = np.copy(self.alpha), np.copy(self.beta)
            change = (np.random.uniform()-0.5)*np.random.uniform()*(1-self.index/self.no_temps)**self.lam
            if random_i >= self.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - self.alpha.shape[0]] = new_beta_v[random_i - self.alpha.shape[0]] + change if np.abs(new_beta_v[random_i - self.alpha.shape[0]] + change) < np.pi else new_beta_v[random_i - self.alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if np.abs(new_alpha_v[random_i]+change) < np.pi else new_alpha_v[random_i]-change
            
            # Generate neighbor conformations and energy values
            new_conformation = self.get_conformation(new_alpha_v, new_beta_v)
            new_energy = self.get_energy(new_alpha_v, new_beta_v)
            energy_change = new_energy-energy
            boltzmen[i] = np.exp(-np.clip(energy_change/self.temp, -500, 500))
            
            # Compare via the Boltzmann criterion
            if energy_change < 0 or boltzmen[i] > rannums[i]:
                 self.alpha = new_alpha_v
                 self.beta = new_beta_v
                 conformation = new_conformation
                 energy = new_energy
                 accepts[i] = 1
            else: 
                rejects[i] = 1
        num_accepts = np.sum(accepts)
        num_rejects = np.sum(rejects)
        
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
    
    def get_child(self, mate):
        """ 
        :param mate: Another annealer
        :returns daughter: A daughter annealer 
        """
        # Determine whether a mutation will take place
        mutation = True if np.random.randint(1, 100) == 42 else False # 42 is the answer to everything
        
        # Compute the daughter characteristics
        lam_d = (2*self.lam*mate.lam)/(self.lam + mate.lam) # Harmonic mean
        rano_a, rano_b = np.random.randint(0, self.alpha.shape[0]-1), np.random.randint(0, mate.beta.shape[0]-1)
        
        # Slice the daughters
        alpha_d =  np.concatenate((self.alpha[0:rano_a], mate.alpha[rano_a:])) 
        beta_d = np.concatenate((self.beta[0:rano_b], mate.beta[rano_b:]))   
        
        # Perturb the components of bond and torsion angle vectors in the case of a mutation
        if mutation: 
            change = (np.random.uniform()-0.5)*np.random.uniform()*(1-self.index/self.no_temps)**lam_d
            random_i = np.random.randint(0, self.alpha.shape[0]+self.beta.shape[0])
            if random_i >= self.alpha.shape[0]:
                # Prevents out of bound errors
                beta_d[random_i - self.alpha.shape[0]] = beta_d[random_i - self.alpha.shape[0]] + change if np.abs(beta_d[random_i - self.alpha.shape[0]] + change) < np.pi else beta_d[random_i - self.alpha.shape[0]]-change
            else:
                # Prevents out of bound errors
                alpha_d[random_i] = alpha_d[random_i]+change if np.abs(alpha_d[random_i]+change) < np.pi else alpha_d[random_i]-change
        
        # Return the child as an Annealer object
        childargs = { 
            'temp': self.temp,
            'lam': lam_d, 
            'ml': self.ml,
            'residues': self.residues,
            'no_temps': self.no_temps,
            'index': self.index,
            'alpha': alpha_d,
            'beta': beta_d,
            'device_id': self.device_id
        }
        return Annealer(**childargs)

class annealerOutput: 
    annealer: Annealer
    energies: ArrayLike
    conformations: ArrayLike
    boltzmen: ArrayLike
    alpha: ArrayLike
    beta: ArrayLike
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

    def __init__(self, annealer: Annealer, energies:ArrayLike, conformations:ArrayLike, boltzmen:ArrayLike, alpha:ArrayLike, beta:ArrayLike, num_accepts: float, num_rejects: float): 
        self.annealer = annealer
        self.energies = energies
        self.conformations = conformations
        self.boltzmen = boltzmen
        self.alpha = alpha
        self.beta = beta
        self.num_accepts = num_accepts
        self.num_rejects = num_rejects

# Genetic Algorithm Handler 
class GeneticAnnealer: 
    num_iterations: int 
    temp: float
    num_annealers: int 
    ml: int 
    quality_factor: float
    lam: float
    residues: str
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
    def __init__(self, num_iterations, temp, num_annealers, ml, quality_factor, lam, residues, index):
        self.num_annealers = num_annealers
        self.num_iterations = num_iterations
        self.temp = temp 
        self.ml = ml 
        self.quality_factor = quality_factor
        self.lam = lam 
        self.index = index
        self.residues = np.array([1 if char == 'A' else 0 for char in residues]) 

    def wrapper(self, cpu_id: int, temp: float, annealerObject: Annealer):
        """ 
        :param cpu_id: The ID of the core 
        :param temp: The current annealing temperature 
        :param annealerObject: A preinitialized annealer that is updated to the current temperature.
        :returns results: A list of annealerOutput objects
        """
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
        energy_std = np.std(runData.energies)
        acceptance_mean = np.mean(runData.run_accepts)
        
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
        J_i = runData.energies[:, -1]
        fit = (J_i - np.min(J_i))/(np.max(J_i - np.min(J_i))) # [0,1]
        # Sort in descending order of fitness
        cmpd_arr = sorted(list(zip(fit.tolist(), runData.annealers)), key=lambda x: x[0], reverse=True)
        fit, annealers = list(zip(*cmpd_arr))
        
        # Stabilising selection
        
        parents = annealers[:len(fit)//4-1]+annealers[-len(fit)//4:]
        # Compute the weighted probability list based on the geometric progression
        probability_ratios = 2**np.arange(0, len(parents), 1)
        probability_ratios = probability_ratios.tolist()[::-1]
        weighted_parents = [item for item, count in zip(parents,probability_ratios) for _ in range(count)]

        # Create the new population 
        while len(population) < len(annealers):
            p1 = weighted_parents[np.random.randint(0, len(weighted_parents)-1)] 
            p2 = weighted_parents[np.random.randint(0, len(weighted_parents)-1)]
            child = p1.get_child(p2)
            population.append(child) 
        return population
          
    def run(self, annealers: list[Annealer]= None)->annealerOutput: 
        if annealers is None:
            annealers = [] 
            for i in range(self.num_annealers): 
                args = { 
                'temp': self.temp, 
                'lam': (self.lam-0.5) + np.random.uniform(), # [lam-1, lam+1] 
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
            run_energies = np.zeros((self.num_annealers, self.ml)) 
            run_conformations = np.zeros((self.num_annealers, self.ml, self.residues.shape[0], 3)) 
            run_boltzmen = np.zeros((self.num_annealers, self.ml)) 
            run_optimal_alphas = np.zeros((self.num_annealers, self.residues.shape[0]-2)) 
            run_optimal_betas = np.zeros((self.num_annealers, self.residues.shape[0]-3))
            run_accepts = np.zeros(self.num_annealers)
            run_rejects = np.zeros(self.num_annealers)
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
        temps = np.zeros(self.num_iterations,)
        initial_run = self.run()
        outputs.append(initial_run.to_dict())
        population_energies = np.zeros((self.num_annealers ,self.ml*self.num_iterations))
        new_annealers = self.crossover(initial_run) 
        self.temp_updater(self.temp, initial_run)
        for i in range(self.num_iterations):
            temps[i]=(self.temp)
            run = self.run(new_annealers)
            self.index+=1
            outputs.append(run.to_dict())
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
    
def artificial_protein(n):
    S = [np.array([1]), np.array([0])]
    
    for i in range(2, n + 1):
        concatenated = np.concatenate((S[i-2], S[i-1]))
        S.append(concatenated)
    nums = S[n]
    result = ''.join(['A' if element == 1 else 'B' for element in nums])
    return result


