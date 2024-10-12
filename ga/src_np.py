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

# Problem statement Classes and Functions
 
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
        # In the context here, optimal energy is of this annealer, not of the overall optimization 
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

            if self.step == 0: 
                x.alpha = new_alpha
                x.beta = new_beta 
                x.energy = new_energy
                


    def optimize(self): 
        # Initializations 
        if self.population_size > multiprocessing.cpu_count(): raise ValueError(f'That population exceeds the number of available cores.')
        # Generate the coefficient matrix using the Kronecker product
        
        # Generate the required initial population based on whether an initial genotype is provided or not         
        if self.init_alpha is None or self.init_beta is None:
            
            # Generate initial alpha and beta chromosome batches
            alphas = -np.pi + np.random.uniform(low=0.0, high=1.0, size=(self.population_size, self.protein.shape[0]-2))*2*np.pi
            betas = -np.pi + np.random.uniform(low=0.0, high=1.0, size=(self.population_size, self.protein.shape[0]-3))*2*np.pi
            
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

        for i in range(population): 
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
                    best, better = subset_genotypes[np.argsort(subset_fitness)[:2][0]], subset_genotypes[np.argsort(subset_fitness)[:2][1]] 
                    
                    # Produce a child from these parents 
                    ran_alpha = np.random.randint(low=0, high=best.alpha.shape[0])
                    ran_beta = np.random.randint(low=0, high=best.beta.shape[0])
                    
                    child_alpha = np.concatenate((best.alpha[:ran_alpha], better.alpha[ran_alpha:])) 
                    child_beta = np.concatenate((best.beta[:ran_beta], better.beta[ran_beta:]))
                    
                    new_population.append(genotype(child_alpha, child_beta, elite=False))

        for i in range(self.num_iterations):
            self.step = i
            
        

test_protein = np.array([1,0,1,0,1,1])
x = optimizer(100, test_protein, 20)
x.optimize()
    
