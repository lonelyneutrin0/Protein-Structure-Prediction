import numpy as np 
from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.random import randint 
import time 

@dataclass
class AnnealerOutput:
    alpha: ArrayLike
    beta: ArrayLike 
    energies: ArrayLike
    conformations: ArrayLike
    optimal_energy: float 
    optimal_conformation: ArrayLike
    p_num_accepts: ArrayLike
    p_num_rejects: ArrayLike
    p_inv_temps: ArrayLike
    residues: ArrayLike

    def to_dict(self): 
        return { 
            'alpha': self.alpha, 
            'beta': self.beta, 
            'energies': self.energies, 
            'conformations': self.conformations, 
            'optimal_energy': self.optimal_energy, 
            'optimal_conformation': self.optimal_conformation, 
            'p_num_accepts': self.p_num_accepts, 
            'p_num_rejects': self.p_num_rejects, 
            'residues': self.residues
        }
def get_coefficient(
    residues: ArrayLike
) -> float: 
    
    """
    
    Return the coefficient determining the strength of interactions between two residues
    If both are 1, then it returns 1 
    If one is 0, it returns 0.5 
    If both are 0, it returns 0.5
    
    :param i: First residue 
    :param j: Second residue  
    """
    coeff = residues[:, np.newaxis]*residues 
    coeff[coeff == 0] = 0.5
    return coeff


def get_conformation(
    size: int, 
    alpha: ArrayLike, 
    beta: ArrayLike,
) -> ArrayLike: 
    
    """
    
    It returns a vector of residue positions based on the bond and torsion angles
    
    :param size: Number of residues 
    :param alphas: Vector of bond angles 
    :param betas: Vector of torsion angles 
    :raises ValueError: If the size is less than 3
    
    """
    
    if(size < 3): 
        raise ValueError(f'Make sure your number of residues is greater than 3')
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    positions = np.zeros((size, 3), dtype=np.float64)
    positions[0] = np.array([0,0,0])
    positions[1] = np.array([0,1,0])
    positions[2] = positions[1] + np.array([cos_alpha[0], sin_alpha[0], 0])
    for i in range(size-3): 
        positions[i+3] = positions[i+2]+np.array([cos_alpha[i+1]*cos_beta[i], sin_alpha[i+1]*cos_beta[i], sin_beta[i]]) 
    return positions

def get_energy(
    alpha: ArrayLike, 
    beta: ArrayLike, 
    residues: ArrayLike, 
    conformation: ArrayLike, 
    coeff: ArrayLike,
    k_1: float=-1.0, 
    k_2: float=0.5
) -> float: 
    
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
    
    backbone_bending_energy = -k_1 * np.sum(np.cos(alpha))
    torsion_energy = -k_2 * np.sum(np.cos(beta))
    
    # Computation of the distance matrix using a vector of residue positions. 
    distance_matrix = np.linalg.norm(conformation[:, np.newaxis] - conformation, axis=-1)
    np.fill_diagonal(distance_matrix, np.inf)
    distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
    total_energy = backbone_bending_energy + torsion_energy + np.sum(np.triu(4*distance_matrix*coeff, k=2))
    return total_energy

def n_annealer(
        residues: ArrayLike, 
        start_temp: float,  
        end_temp: float,
        gamma: float, 
        lam: float,
        ml: int,
        init_alpha: ArrayLike = None, 
        init_beta: ArrayLike = None,
        k_1: float = -1.0,
        k_2: float = 0.5,
) -> AnnealerOutput: 
    
    """ 
    
    Algorithm that performs simulated annealing paired with stochastic tunneling 
    

    :param residues: A N bit vector containing information about the hydrophobicity of each residue {-1, 1} 
    :param start_temp: The temperature annealing starts at 
    :param end_temp: The temperature annealing ends at
    :param gamma: The cooling coefficient- T_k+1 = gamma*T_k
    :param lam: The tuned constant representing heterogeneous degree 
    :param ml: The markov chain length, number of times to find a new neighbor at a given temperature.
    :param init_alpha: An initial bond angle vector to start annealing with [-π, π]
    :param init_beta: An initial torsion angle vector to start annealing with [-π, π]
    :param k_1: Weight parameter for backbone bending energy. Default = 1.0 
    :param k_2: Weight parameter for torsion energy. Default = 0.5
    :return: AnnealingOutput
    :raises ValueError: if alpha is not of size N-2 
                        if beta is not of size N-3
    
    """
    
    if init_alpha is None: 
        alpha_v = -np.pi + 2*np.pi*np.random.uniform(size=(residues.shape[0]-2),low=0,high=1)
    else: 
        alpha_v = init_alpha
    
    if init_beta is None: 
        beta_v = -np.pi + 2*np.pi*np.random.uniform(low=0,high=1, size=(residues.shape[0]-3))
    else: 
        beta_v = init_beta
    
    if(alpha_v.shape[0] !=  residues.shape[0]-2 or beta_v.shape[0] != residues.shape[0]-3): 
        raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
    
    # Initializations 

    # Force integer number of iterations
    num_iterations = (int)(np.log10(end_temp/start_temp)/np.log10(gamma))
    
    inv_temps = 1/start_temp*np.power(1/gamma, np.arange(num_iterations))
    energies = np.zeros(num_iterations)
    conformations = np.zeros((num_iterations, residues.shape[0], 3))
    accepts = np.zeros(num_iterations)
    rejects = np.zeros(num_iterations)
    conformation = get_conformation(
        alpha=alpha_v, 
        beta=beta_v, 
        size=residues.shape[0]
    )
    coeff = get_coefficient(residues)
    args = {
        'alpha':alpha_v,
        'beta': beta_v,
        'residues': residues,
        'conformation': conformation, 
        'coeff': coeff
    }
    
    energy = get_energy(**args)
    random_numbers = np.random.random(num_iterations)
    
    # Optimal Values
    optimal_energy = energy 
    optimal_conformation = conformation
    
    for step in range(num_iterations): 
        # Log the current values
        energies[step] = energy
        conformations[step] = conformation
        past = time.perf_counter()
        for i in range(ml):
            # Modify bond vectors and update the arguments of the energy function accordingly.
            new_alpha_v, new_beta_v = np.copy(alpha_v), np.copy(beta_v)
            random_i = randint(alpha_v.shape[0]+beta_v.shape[0])
            change = (np.random.uniform(0,1)-0.5)*np.random.uniform(0,1)*(1-step/num_iterations)**lam
            if random_i >= alpha_v.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - alpha_v.shape[0]] = new_beta_v[random_i - alpha_v.shape[0]] + change if np.abs(new_beta_v[random_i - alpha_v.shape[0]] + change) < np.pi else new_beta_v[random_i - alpha_v.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if np.abs(new_alpha_v[random_i]+change) < np.pi else new_alpha_v[random_i]-change
            
            new_conformation = get_conformation(
                alpha=new_alpha_v, 
                beta=new_beta_v, 
                size=residues.shape[0]
            )
        
            args['alpha'], args['beta'], args['conformation'] = new_alpha_v, new_beta_v, new_conformation
            # Calculate the changes in energy level
            new_energy = get_energy(**args) 
            energy_change = new_energy - energy
            
            # keep track of the optimal values
            if(new_energy < optimal_energy): 
                optimal_energy = new_energy
                optimal_conformation = new_conformation   
            
            # accept or reject the new vectors using the metropolis condition and boltzmann factor
            if energy_change < 0 or random_numbers[step] < np.exp(-inv_temps[step]*energy_change): 
                alpha_v = new_alpha_v   
                beta_v = new_beta_v
                conformation = new_conformation
                energy = new_energy
                accepts[step] = 1
            else: 
                rejects[step] = 1
        print(f'{step+1} Iteration, {num_iterations-step-1} Remaining \n Time Elapsed: {time.perf_counter()-past}')
    num_accepts, num_rejects = np.cumsum(accepts, axis=0), np.cumsum(rejects, axis=0)
    
    annealing_attributes = { 
        'alpha': alpha_v,
        'beta': beta_v,
        'energies': energies,
        'conformations': conformations,
        'optimal_energy': optimal_energy,
        'optimal_conformation': optimal_conformation,
        'p_num_accepts': num_accepts,
        'p_num_rejects': num_rejects,
        'p_inv_temps': inv_temps,
        'residues': residues
    }
    
    return AnnealerOutput(**annealing_attributes)

def artificial_protein(n):
    S = [np.array([1]), np.array([0])]
    
    for i in range(2, n + 1):
        concatenated = np.concatenate((S[i-2], S[i-1]))
        S.append(concatenated)
    return S[n]
