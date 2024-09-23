# pytorch implementation 
import torch 
import numpy as np
from numpy.random import randint
from dataclasses import dataclass
from numpy.typing import ArrayLike
import random
import time 
import math
@dataclass 
class AnnealerOutput: 
    alpha: torch.tensor
    beta: torch.tensor 
    energies: torch.tensor
    conformations: torch.tensor
    optimal_energy: float 
    optimal_conformation: torch.tensor
    p_num_accepts: torch.tensor
    p_num_rejects: torch.tensor
    p_inv_temps: torch.tensor
    residues: torch.tensor

    """

    Container class for the output of the annealing algorithm 
    returns: 
    alpha: Optimal bond angle vector 
    beta: Optimal torsion angle vector 
    energies: The energy series of the annealing process 
    conformations: The conformation series of the annealing process 
    optimal_energy: The minimum energy obtained in the annealing process 
    optimal_conformation: The conformation corresponding to the optimal energy state 
    p_num_accepts: The number of accepted neighbors
    p_num_rejects: The number of rejected neighbors 
    residues: The array of amino acid residues  

    """

def get_coefficient(x: torch.tensor) -> torch.tensor:
    """
    
    Return the coefficient determining the strength of interactions between two residues
    If both are 1, then it returns 1 
    If one is 0, it returns 0.5 
    If both are 0, it returns 0.5
    
    :param i: First residue 
    :param j: Second residue 

    """
    coeff_matrix = x[:, None] * x 
    coeff_matrix[coeff_matrix == 0] = 0.5
    return coeff_matrix

def get_conformation( 
    size: int, 
    alpha: torch.tensor, 
    beta: torch.tensor,
) -> torch.tensor: 
    
    """
    
    It returns a vector of residue positions based on the bond and torsion angles
    
    :param size: Number of residues 
    :param alphas: Vector of bond angles 
    :param betas: Vector of torsion angles 
    :raises ValueError: If the size is less than 3
    
    """
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

def get_energy(
    alpha: torch.tensor, 
    beta:  torch.tensor, 
    residues:  torch.tensor, 
    conformation:  torch.tensor, 
    coeff: torch.tensor,
    k_1: float=-1.0, 
    k_2: float=0.5
) -> float: 
    """ 
    
    Protein Conformation Energy Function 
    
    :param alpha: Bond angle vector
    :param beta: Torsion angle vector
    :param residues: Array of the residue variables 
    :param conformation: A matrix of residue positions
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy
    :returns: float containing the potential energy value
    
    The energy is split into backbone bending energy, torsion energy and energy based on the hydrophobic interactions of residues.
    The third energy is computed using matrices and the required mask. 
    
    """
    
    # Cast to torch.tensor
    backbone_bending_energy = -k_1*torch.sum(torch.cos(alpha))
    torsion_energy = -k_2*torch.sum(torch.cos(beta))
    distance_matrix = torch.linalg.norm(conformation[:, None] - conformation, dim=-1)
    distance_matrix.fill_diagonal_(torch.inf)
    distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
    total_energy = backbone_bending_energy + torsion_energy + torch.sum(torch.triu(4*distance_matrix*coeff, diagonal=2))
    return total_energy

def t_annealer(
    residues:  torch.tensor, 
    start_temp: float,  
    end_temp: float,
    gamma: float, 
    lam: float,
    ml: int,
    init_alpha:  torch.tensor = None, 
    init_beta:  torch.tensor = None,
    k_1: float = -1.0,
    k_2: float = 0.5,
) -> torch.Tensor: 
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
        alpha_v = -torch.pi + 2*torch.pi*torch.rand(residues.shape[0]-2)
    else: 
        alpha_v = init_alpha
    
    if init_beta is None: 
        beta_v = -torch.pi + 2*torch.pi*torch.rand(residues.shape[0]-3)
    else: 
        beta_v = init_beta

    if(alpha_v.shape[0] !=  residues.shape[0]-2 or beta_v.shape[0] != residues.shape[0]-3): 
        raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
    
    num_iterations = (int)(math.log10(end_temp/start_temp)/math.log10(gamma))
    iterations = torch.arange(num_iterations)
    inv_temps = 1/start_temp*(1/gamma)**iterations
    energies = torch.zeros(num_iterations)
    conformations = torch.zeros((num_iterations, residues.shape[0], 3))
    accepts = torch.zeros(num_iterations)
    rejects = torch.zeros(num_iterations)
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
    random_numbers = torch.from_numpy(np.random.random(num_iterations))
    optimal_energy = energy 
    optimal_conformation = conformation
    for step in range(num_iterations): 
        # Log the current values
        energies[step] = energy
        conformations[step] = conformation
        past = time.perf_counter()
        for i in range(ml):
            # Modify bond vectors and update the arguments of the energy function accordingly.
            random_i = randint(alpha_v.shape[0]+beta_v.shape[0])
            new_alpha_v, new_beta_v = torch.clone(alpha_v), torch.clone(beta_v)
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
            
            # accept or reject the new vectors using the metropolis condition
            if energy_change < 0 or random_numbers[step] < torch.exp(-inv_temps[step]*energy_change): 
                alpha_v = new_alpha_v   
                beta_v = new_beta_v
                conformation = new_conformation
                energy = new_energy
                accepts[step] = 1
            else: 
                rejects[step] = 1
        print(f'{step+1} Iteration, {num_iterations-step-1} Remaining \n Time Elapsed: {time.perf_counter()-past}')
    num_accepts, num_rejects = torch.cumsum(accepts, axis=0), torch.cumsum(rejects, axis=0)
    
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
    S = [torch.array([1]), torch.array([0])]
    
    for i in range(2, n + 1):
        concatenated = torch.concatenate((S[i-2], S[i-1]))
        S.append(concatenated)
    return S[n]



