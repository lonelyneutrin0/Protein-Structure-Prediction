import numpy as np 
from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.random import randint 

@dataclass
class AnnealingOutput:
    optimal_alpha: ArrayLike 
    optimal_beta: ArrayLike 
    optimal_energies: ArrayLike
    optimal_conformations: ArrayLike
    optimalest_energy: float 
    optimalest_conformation: ArrayLike
    param_num_accepts: ArrayLike
    param_num_rejects: ArrayLike
    param_inv_temps: ArrayLike


def get_coefficient(
    param_i: bool, 
    param_j: bool, 
) -> float: 
    
    """
    
    Return the coefficient determining the strength of interactions between two residues
    If both are 1, then it returns 1 
    If one is 0, it returns 0.5 
    If both are 0, it returns 0.5

    :param i: First residue 
    :param j: Second residue  
    """

    if(param_i + param_j == 2): 
        return 1
    return 0.5

# Vectorize the function
get_coefficient = np.vectorize(get_coefficient)

def get_conformation(
    param_size: int,
    param_alphas: ArrayLike, 
    param_betas: ArrayLike,
) -> ArrayLike: 
    
    """
    
    It returns a vector of residue positions based on the bond and torsion angles
    
    :param size: Number of residues 
    :param alphas: Vector of bond angles 
    :param betas: Vector of torsion angles 
    :raises ValueError: If the size is less than 3

    """

    if(param_size < 3): 
        raise ValueError(f'Make sure your number of residues is greater than 3')
    cos_alpha = np.cos(param_alphas)
    cos_beta = np.cos(param_betas)
    sin_alpha = np.sin(param_alphas)
    sin_beta = np.sin(param_betas)
    positions = np.zeros((param_size, 3), dtype=float)
    positions[0] = np.array([0,0,0])
    positions[1] = np.array([0,1,0])
    positions[2] = positions[1] + np.array([cos_alpha[0], sin_alpha[0], 0])
    for i in range(param_size-3): 
        positions[i+3] = positions[i+2]+np.array([cos_alpha[i+1]*cos_beta[i], sin_alpha[i+1]*cos_beta[i], sin_beta[i]]) 
    
    return positions

def get_energy(
        param_alpha: ArrayLike, 
        param_beta: ArrayLike, 
        param_residues: ArrayLike, 
        param_conformation: ArrayLike,
        param_k_1: float = 1.0, 
        param_k_2: float = 0.5
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
    
    backbone_bending_energy = -(param_k_1**param_alpha.shape[0]) * np.sum(np.cos(param_alpha))
    torsion_energy = -(param_k_2**param_beta.shape[0]) * np.sum(np.cos(param_beta))
    
    # Computation of the distance matrix using a vector of residue positions. 
    distance_matrix = (np.linalg.norm(param_conformation[:, np.newaxis] - param_conformation, axis=-1))
    np.fill_diagonal(distance_matrix, np.inf)
    
    distance_matrix = distance_matrix**(-6) - distance_matrix**(-12)
    total_energy = backbone_bending_energy + torsion_energy + np.sum(np.triu(4*distance_matrix*get_coefficient(param_residues[:, np.newaxis], param_residues), k=2))
    return total_energy

def annealer(
        residues: ArrayLike, 
        start_temp: float,  
        end_temp: float,
        gamma: float, 
        lam: float,
        init_alpha: ArrayLike = None, 
        init_beta: ArrayLike = None,
        k_1: float = 1.0,
        k_2: float = 0.5,
) -> AnnealingOutput: 
    
    """ 
    
    Algorithm that performs simulated annealing paired with stochastic tunneling 


    :param residues: A N bit vector containing information about the hydrophobicity of each residue {-1, 1} 
    :param start_temp: The temperature annealing starts at 
    :param end_temp: The temperature annealing ends at
    :param gamma: The cooling coefficient- T_k+1 = gamma*T_k
    :param lam: The tuned constant representing heterogeneous degree 
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
        param_alphas=alpha_v, 
        param_betas=beta_v, 
        param_size=residues.shape[0]
    )
    args = {
        'param_alpha':alpha_v,
        'param_beta': beta_v,
        'param_residues': residues,
        'param_conformation': conformation
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

        # Modify bond vectors and update the arguments of the energy function accordingly.
        random_i = randint(alpha_v.shape[0]+beta_v.shape[0])
        new_alpha_v, new_beta_v = np.copy(alpha_v), np.copy(beta_v)

        if random_i >= alpha_v.shape[0]: 
            new_beta_v[random_i - alpha_v.shape[0]] += (np.random.uniform(0,1)-0.5)*np.random.uniform(0,1)*(1-step/num_iterations)**lam
        else: 
            new_alpha_v[random_i] += (np.random.uniform(0,1)-0.5)*np.random.uniform(0,1)*(1-step/num_iterations)**lam
        
        new_conformation = get_conformation(
            param_alphas=new_alpha_v, 
            param_betas=new_beta_v, 
            param_size=residues.shape[0]
        )
        
        args['param_alpha'], args['param_beta'], args['param_conformation'] = new_alpha_v, new_beta_v, new_conformation
        # Calculate the changes in energy level
        new_energy = get_energy(**args) 
        energy_change = new_energy - energy
        
        
        # get the boltzmann factor corresponding to the change
        boltzmann_factor = np.exp(-inv_temps[step]*energy_change)
        
        # keep track of the optimal values
        if(new_energy < optimal_energy): 
            optimal_energy = new_energy
            optimal_conformation = new_conformation   
        
        # accept or reject the new vectors using the metropolis condition
        if energy_change < 0 or random_numbers[step] < boltzmann_factor: 
            alpha_v = new_alpha_v   
            beta_v = new_beta_v
            conformation = new_conformation
            energy += energy_change
            accepts[step] = 1
        else: 
            rejects[step] = 1
    
    num_accepts, num_rejects = np.cumsum(accepts, axis=0), np.cumsum(rejects, axis=0)
    
    annealing_attributes = { 
        'optimal_alpha': alpha_v,
        'optimal_beta': beta_v,
        'optimal_energies': energies,
        'optimal_conformations': conformations,
        'optimalest_energy': optimal_energy,
        'optimalest_conformation': optimal_conformation,
        'param_num_accepts': num_accepts,
        'param_num_rejects': num_rejects,
        'param_inv_temps': inv_temps,
    }
    
    return AnnealingOutput(**annealing_attributes)