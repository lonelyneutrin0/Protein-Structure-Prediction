# Generate a population of solutions at T_0
# Compute the fitness value of each of the solutions
# Crossover with a probability proportional to the fitness 
# Repeat this N amount of times
#   Mutate randomly using simulated annealing 
# Update the old generation with the new generation, using elitism and preserving the best members of the old population 

from dataclasses import dataclass
import numpy as np 
import multiprocessing
import matplotlib.pyplot as plt 
boltzmann = 1.380649e-23

# Problem statement Classes and Functions
 
def artificial_protein(n):
    S = [np.array([1]), np.array([0])]

    for i in range(2, n + 1):
        concatenated = np.concatenate((S[i-2], S[i-1]))
        S.append(concatenated)
    nums = S[n]
    return nums

def get_conformation(
    size: int, 
    alphas: np.ndarray, 
    betas: np.ndarray,
    population_size: int=1
) -> np.ndarray: 
    
    """
    
    It returns a vector of residue positions based on the bond and torsion angles of a population
    
    :param size: Number of residues 
    :param alphas: Vector of bond angles 
    :param betas: Vector of torsion angles 
    :raises ValueError: If the size is less than 3
    
    """

    if(size < 3): 
        raise ValueError(f'Make sure your number of residues is greater than 3')
    
    # If the population size is 1, convert size <4> to <1,4>
    if population_size == 1: 
        alphas = alphas[np.newaxis, :]
        betas = betas[np.newaxis, :]
    
    cos_alpha = np.cos(alphas)
    cos_beta = np.cos(betas)
    sin_alpha = np.sin(alphas)
    sin_beta = np.sin(betas)
    positions = np.zeros((population_size, size, 3), dtype=np.float64)
    positions[:, 0] = np.array([0,0,0])
    positions[:, 1] = np.array([0,1,0])
    positions[:, 2, 0] = positions[:, 1, 0] + cos_alpha[:, 0] 
    positions[:, 2, 1] = positions[:, 1, 1] + sin_alpha[:, 0]
    for i in range(size-3): 
        positions[:, i+3, 0] = positions[:, i+2, 0] +  cos_alpha[:, i+1]*cos_beta[:, i]
        positions[:, i+3, 1] = positions[:, i+2, 1] + sin_alpha[:, i+1]*cos_beta[:, i]
        positions[:, i+3, 2] = positions[:, i+2, 2] + sin_beta[:, i] 
    return positions if positions.shape[0] != 1 else np.squeeze(positions) # Ensures that the output is of the proper dimension

def get_energy(
    alphas: np.ndarray, 
    betas: np.ndarray, 
    protein: np.ndarray, 
    conformations: np.ndarray, 
    coeff: np.ndarray,
    population_size: int=1,
    k_1: float=-1.0, 
    k_2: float=0.5
) -> np.ndarray: 
    
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
    
    # If the population size is 1, convert size <4> to <1,4>
    if population_size == 1: 
        alphas = alphas[np.newaxis, :]
        betas = betas[np.newaxis, :]
        conformations = conformations[np.newaxis, :, :]

    backbone_bending_energy = -k_1 * np.sum(np.cos(alphas), axis=-1)
    torsion_energy = -k_2 * np.sum(np.cos(betas), axis=-1)
    
    # Computation of the distance matrix using a vector of residue positions. 
    distance_matrix = np.linalg.norm(conformations[:, :, None, :] - conformations[:, None, :, :], axis=-1)
    protein_indices = np.arange(distance_matrix.shape[1])
    distance_matrix[:, protein_indices, protein_indices] = np.inf
    distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
    population_indices = np.arange(population_size)
    total_energy = np.zeros(population_size)
    total_energy[population_indices] = backbone_bending_energy[population_indices] + torsion_energy[population_indices] + np.sum((np.triu(4*distance_matrix[population_indices]*coeff, k=2)), axis=(1,2))
    return total_energy if population_size > 1 else np.sum(total_energy)

# Optimizer Classes and Functions
@dataclass 
class annealerOutput:
    energies: np.ndarray
    num_accepts: int
    num_rejects: int 
    optimal_genotype: np.ndarray
    
    """
    A container class for each mutation cycle 

    :param energies: The energy series
    :param num_accepts: Number of accepted neighbors 
    :param num_rejects: Number of rejected neighbors
    :param optimal_genotype: The optimal genotype 
    """
@dataclass 
class genotype: 
    alpha: np.ndarray
    beta: np.ndarray
    energy: float=0 
    conformation: np.ndarray=None
    elite: bool = False
    
    """
    A class representing the genotype of each candidate solution
    
    :param alpha: The bond angle vector 
    :param beta: The torsion angle vector
    :param fitness: The fitness of the candidate solution 
    :param crossover_probability: The probability of it being involved in crossover
    :param elite: Whether it is an elite member of the population 
    """

@dataclass 
class solution: 
    sol: genotype 
    energies: np.ndarray
    fitnesses: np.ndarray
    elites: list[genotype]
    
    """ 
    A container class for the solution 
    
    :param sol: The fittest individual's genotype 
    :param energies: The energy series of the optimization 
    :param fitnesses: The fitness series of the optimization 
    :param elites: The elites of each generation in the population 
    """

class optimizer: 
    T_0: float
    protein: np.ndarray
    population_size: int
    p_c: float = 0.8 
    ml: int=10000
    num_iterations: int=3000
    lam: float=3.0 
    init_alpha: np.ndarray=None
    init_beta: np.ndarray=None
    step:int=0
    
    """ 
    An input class for hyperparameters 
    
    :param T_0: Initial temperature 
    :param protein: The input protein sequence 
    :param population_size: The number of candidate solutions per population.
    :param p_c: The probability of crossover
    :param ml: The markov chain length 
    :param lam: The heterogenous tuned constant 
    :param init_alpha: An initial alpha chromosome 
    :param init_beta: An initial beta chromosome 
    :param step: The current iteration of the optimizer
    """
    
    def __init__(self, T_0, protein, population_size, p_c=0.8, ml=10000, num_iterations=3000, lam=3.0, init_alpha=None, init_beta=None, step=0):
        self.T_0 = T_0
        self.protein = protein 
        self.population_size = population_size
        self.p_c = p_c
        self.ml = ml
        self.num_iterations = num_iterations
        self.lam = lam 
        self.init_alpha = init_alpha 
        self.init_beta = init_beta
        self.coeff = 0.5*np.kron(self.protein, self.protein).reshape(self.protein.shape[0], self.protein.shape[0]) + 0.5
        self.step = step
        
    def anneal(self, x: genotype): 
        # Optimal energy is of this annealer, not of the overall optimization 
        optimal_alpha = x.alpha
        optimal_beta = x.beta
        optimal_energy = x.energy
        optimal_conformation = x.conformation
        for j in range(self.ml): 
            # Mutation 
            new_alpha = x.alpha.copy()
            new_beta = x.beta.copy()
            ran_i = np.random.randint(x.alpha.shape[0]+x.beta.shape[0])
            change = (np.random.uniform()-0.5)*np.random.uniform()*(1-self.step/self.num_iterations)**self.lam

            if ran_i >= x.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta[ran_i - new_alpha.shape[0]] = new_beta[ran_i- new_alpha.shape[0]] + change if np.abs(new_beta[ran_i - new_alpha.shape[0]] + change) < np.pi else new_beta[ran_i- new_alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha[ran_i] = new_alpha[ran_i]+change if np.abs(new_alpha[ran_i]+change) < np.pi else new_alpha[ran_i]-change
        
            # Determine the new genotype characteristics
            new_conformation = get_conformation(self.protein.shape[0], new_alpha, new_beta)
            new_energy = get_energy(new_alpha, new_beta, self.protein, new_conformation, self.coeff)
            
            # Always accept the initial few states
            if self.step in range(10): 
                x.alpha = new_alpha
                x.beta = new_beta 
                x.energy = new_energy
                continue 

            # Use the Boltzmann criterion to determine whether a neighbor is accepted
            if np.exp((x.energy-new_energy)/(self.T_0)) > np.random.uniform(): 
                x.alpha = new_alpha
                x.beta = new_beta 
                x.energy = new_energy

            # Update the optimal conformations 
            if new_energy < optimal_energy: 
                optimal_alpha = new_alpha
                optimal_beta = new_beta
                optimal_energy = new_energy
                optimal_conformation = new_conformation
        
        return genotype(alpha=optimal_alpha, beta=optimal_beta, energy=optimal_energy, conformation=optimal_conformation)     
    
    def optimize(self): 
        # Initializations 
        if self.population_size > multiprocessing.cpu_count(): raise ValueError(f'That population exceeds the number of available cores.')
        
        # Generate the required initial population based on whether an initial genotype is provided or not         
        if self.init_alpha is None or self.init_beta is None:
            
            # Generate initial alpha and beta chromosome batches
            alphas = np.random.uniform(low=-np.pi, high=np.pi, size=(self.population_size, self.protein.shape[0]-2))
            betas = np.random.uniform(low=-np.pi, high=np.pi, size=(self.population_size, self.protein.shape[0]-3))
            
            # Generate the population 
            population = [genotype(alphas[i], betas[i]) for i in range(self.population_size)]
        
        else: 
            
            # Modify the initial vectors slightly before producing the population to promote genetic diversity 
            alphas = np.tile(self.init_alpha, (self.population_size, 1))+np.eye((self.population_size, self.protein.shape[0]-2))*np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, self.protein.shape[0]-2))
            betas = np.tile(self.init_beta, (self.population_size, 1))+np.eye((self.population_size, self.protein.shape[0]-3))*np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, self.protein.shape[0]-3))
            
            # Make sure the values aren't out of bounds 
            alphas, betas = np.clip(alphas, -np.pi, np.pi), np.clip(betas, -np.pi, np.pi)
            
        # Generate the population, conformations and energies. Update the genotypes accordingly
        population = [genotype(alphas[i], betas[i]) for i in range(self.population_size)]
        population_conformations = get_conformation(self.protein.shape[0], alphas, betas, self.population_size)
        population_energies = get_energy(alphas, betas, self.protein, population_conformations, self.coeff, self.population_size)

        for i in range(len(population)): 
            population[i].conformation = population_conformations[i]
            population[i].energy = population_energies[i]
        
        # Tournament Selection 
        new_population = []
        while len(new_population) < self.population_size: 
            best = None 
            better = None 
            while best is None or better is None: 
                
                # Generate a random subset of the population
                indices = np.random.randint(low=0,high=self.population_size,size= 5)
                subset_genotypes = []
                for i in indices: 
                    subset_genotypes.append(population[i])
                
                # Compute the fitness of those elements
                subset_fitness = population_energies[indices]
                
                if np.random.uniform() > self.p_c: 
                    # Choose the best and better elements of the subset population
                    best, better = subset_genotypes[np.argsort(subset_fitness)[0]], subset_genotypes[np.argsort(subset_fitness)[1]] 
                    
                    # Produce a child from these parents 
                    ran_alpha = np.random.randint(low=0, high=best.alpha.shape[0])
                    ran_beta = np.random.randint(low=0, high=best.beta.shape[0])
                    
                    child_alpha = np.concatenate((best.alpha[:ran_alpha], better.alpha[ran_alpha:])) 
                    child_beta = np.concatenate((best.beta[:ran_beta], better.beta[ran_beta:]))
                    child_conformation = get_conformation(self.protein.shape[0], child_alpha, child_beta)
                    child_energy = get_energy(child_alpha, child_beta, self.protein, child_conformation, self.coeff)
                    new_population.append(genotype(child_alpha, child_beta, child_energy, child_conformation, elite=False))
        
        current_population = new_population          
        
        for i in range(self.num_iterations):
            # Update the step and temperature
            self.step = i
            self.T_0 = 0.99*self.T_0
           
            # Start parallel processing 
            with multiprocessing.Pool(processes=self.population_size) as pool: 
                
                # Anneal each of the genotypes in parallel and collect the mutated population
                mutated_population=pool.map(self.anneal, current_population)
            # Collect the characteristics of the population and perform fitness evaluations 
            mutated_population_conformations = np.zeros((self.population_size, self.protein.shape[0], 3))
            mutated_population_energies = np.zeros(self.population_size)
            
            for i in range(len(mutated_population)): 
                mutated_population_conformations[i] = mutated_population[i].conformation
                mutated_population_energies[i] = mutated_population[i].energy
 
            current_population = []
            # Perform tournament search and produce the new population 
            while len(current_population) < self.population_size: 
                best = None 
                better = None 
                while best is None or better is None: 
                    
                    # Generate a random subset of the population
                    indices = np.random.randint(low=0,high=self.population_size,size= 5)
                    subset_genotypes = []
                    for i in indices: 
                        subset_genotypes.append(mutated_population[i])
                    
                    # Compute the fitness of those elements
                    subset_fitness = mutated_population_energies[indices]
                    
                    if np.random.uniform() > self.p_c: 
                        # Choose the best and better elements of the subset population
                        best, better = subset_genotypes[np.argsort(subset_fitness)[:2][0]], subset_genotypes[np.argsort(subset_fitness)[:2][1]] 
                        
                        # Produce a child from these parents 
                        ran_alpha = np.random.randint(low=0, high=best.alpha.shape[0])
                        ran_beta = np.random.randint(low=0, high=best.beta.shape[0])
                        
                        child_alpha = np.concatenate((best.alpha[:ran_alpha], better.alpha[ran_alpha:])) 
                        child_beta = np.concatenate((best.beta[:ran_beta], better.beta[ran_beta:]))
                        child_conformation = get_conformation(self.protein.shape[0], child_alpha, child_beta)
                        child_energy = get_energy(child_alpha, child_beta, self.protein, child_conformation, self.coeff)
                        current_population.append(genotype(child_alpha, child_beta, child_energy, child_conformation, elite=False))
            print(mutated_population_energies)
        # At the end of the for loop, we have an optimal solution somewhere in the population 
        optimal_population = current_population
        optimal_population_energies = np.zeros(self.population_size)
        optimal_population_conformations = np.zeros((self.population_size, self.protein.shape[0], 3))
        
        # Collect population data
        for i in range(len(optimal_population)): 
            optimal_population_energies[i] = optimal_population[i].energy
            optimal_population_conformations[i] = optimal_population[i].conformation
        
        optimal_genotype = optimal_population[np.argsort(optimal_population_energies)[0]]
        
        return optimal_genotype

test_protein = np.array(artificial_protein(6))
x = optimizer(T_0=1.0, protein=test_protein, population_size=8, ml=1000, num_iterations=100)
if __name__ == '__main__':
    solution = x.optimize()
    print(f'Optimal Energy: {solution.energy}')
