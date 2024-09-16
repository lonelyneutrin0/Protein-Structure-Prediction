import numpy as np 
from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.random import randint 

@dataclass
class AnnealingOutput:
    param_alpha: ArrayLike 
    param_beta: ArrayLike 
    param_energies: ArrayLike
    param_transformed_energies: ArrayLike
    param_stun: int
    param_states_alpha: ArrayLike
    param_o_mins: float
    param_states_beta: ArrayLike
    param_num_accepts: ArrayLike
    param_num_rejects: ArrayLike
    param_inv_temps: ArrayLike


def get_coefficient(
    param_i: float, 
    param_j: float, 
) -> float: 
    
    """
    
    Return the coefficient determining the strength of interactions between two residues
    If both are 1, then it returns 1 
    If one is 0, it returns 0.5 
    If both are 0, it returns 0.5

    :param i: First residue 
    :param j: Second residue  
    """

    if (param_i + param_j == 2): 
        return 1.0
    elif (param_i+param_j == 1): 
        return 0.5 
    elif (param_i+param_j == 0): 
        return 0.5

# Vectorize the function
get_coefficient = np.vectorize(get_coefficient)

def get_residue_positions(
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
        param_residue_positions: ArrayLike,
        param_k_1: float = 1.0, 
        param_k_2: float = 0.5
) -> float: 
    
    """ 
    
    Protein Conformation Energy Function 
    
    :param alpha: Bond angle vector
    :param beta: Torsion angle vector
    :param residues: Array of the residue variables 
    :param distance_matrix: The element d_ij gives the distance between the ith and jth residue
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy
    :returns: float containing the potential energy value
    
    The energy is split into backbone bending energy, torsion energy and energy based on the hydrophobic interactions of residues.
    The third energy is computed using matrices and the required mask. 
    
    """
    
    backbone_bending_energy = np.dot(np.full((param_alpha.shape[0]), -param_k_1), np.cos(param_alpha))
    torsion_energy = np.dot(np.full((param_beta.shape[0]), -param_k_2), np.cos(param_beta))
    
    
    # Computation of the distance matrix using a vector of residue positions. 
    # An identity matrix of desired length is added to prevent division by zero. This does not affect the final calcluation as diagonal values are not included in the mask. 
    distance_matrix = np.linalg.norm(param_residue_positions[np.newaxis, :] - param_residue_positions, axis=-1) + np.eye((param_residue_positions.shape[0], param_residue_positions.shape[0])) 
    distance_matrix = distance_matrix**(-6) - distance_matrix**(-12)
    
    return backbone_bending_energy + torsion_energy + np.sum(np.triu(4*distance_matrix*get_coefficient(param_residues[np.newaxis, :], param_residues)), k=2)

def get_transformed_energy( 
        param_alpha: ArrayLike, 
        param_beta: ArrayLike, 
        param_o_min: float,
        param_residues: ArrayLike,
        param_residue_positions: ArrayLike,
        param_stun: float = 1.0,
        param_k_1: float = 1.0,
        param_k_2: float = 0.5
) -> float:
    
    """ 
    
    Transformed Energy Function
    
    :param alpha: Bond angle vector
    :param beta: Torsion angle vector 
    :param o_min: The smallest objective function value found so far
    :param residues: Array of the residue variables 
    :param residue_positions: Array of the positions of residues
    :param stun: Controls how much the objective function is flattened 
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy 
    :returns: float containing the transformed energy value
    
    """
    
    return 1-np.exp(-param_stun*(get_energy(param_alpha, param_beta, param_residues, param_residue_positions, param_k_1, param_k_2) - param_o_min))

def annealer(
        residues: ArrayLike, 
        num_iterations: int, 
        high_temp: float,  
        low_temp: float, 
        stun: float = 1.0,
        init_alpha: ArrayLike = None, 
        init_beta: ArrayLike = None,
        k_1: float = 1.0,
        k_2: float = 0.5,
) -> AnnealingOutput: 
    
    """ 
    
    Algorithm that performs simulated annealing paired with stochastic tunneling 


    :param residues: A N bit vector containing information about the hydrophobicity of each residue {-1, 1} 
    :param high_temp: The temperature annealing starts at 
    :param low_temp: The temperature annealing ends at 
    :param stun: It determines how much local minima are flattened by the stochastic tunneling transformation 
    :param init_alpha: An initial bond angle vector to start annealing with [-π, π]
    :param init_beta: An initial torsion angle vector to start annealing with [-π, π]
    :param k_1: Weight parameter for backbone bending energy. Default = 1.0 
    :param k_2: Weight parameter for torsion energy. Default = 0.5
    :return: AnnealingOutput
    :raises ValueError: if alpha is not of size N-2 '
                        if beta is not of size N-3
    
    """
    
    if(init_alpha.shape[0] !=  residues.shape[0]-2 or init_beta.shape[0] != residues.shape[0]-3): 
        raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
    
    # Initializations 
    inv_temps = np.linspace(1/high_temp, 1/low_temp, num_iterations)
    energies = np.zeros(num_iterations)
    transformed_energies = np.zeros(num_iterations)
    stun = 1
    states_alpha = np.zeros((num_iterations, residues.shape[0]-2))
    states_beta = np.zeros((num_iterations, residues.shape[0]-3))
    o_min = 0
    accepts = np.zeros(num_iterations)
    rejects = np.zeros(num_iterations)
    
    if init_alpha is None: 
        alpha_v = np.random.uniform(-np.pi, np.pi, residues.shape[0]-2)
    else: 
        alpha_v = init_alpha
    
    if init_beta is None: 
        beta_v = np.random.uniform(-np.pi, np.pi, residues.shape[0]-3)
    else: 
        beta_v = init_beta
    
    residue_positions = get_residue_positions(
        param_alphas=alpha_v, 
        param_betas=beta_v, 
        param_size=residues.shape[0]
    )
    
    
    transformed_energy = get_transformed_energy(
        param_alpha=alpha_v, 
        param_beta=beta_v, 
        param_o_min=o_min,
        param_residues=residues, 
        param_residue_positions=residue_positions,
        param_stun=stun
    )
    
    random_numbers = np.random.random(num_iterations)
    random_modifications_alpha = np.random.uniform(-0.1, 0.1, (alpha_v.size))
    random_modifications_beta = np.random.uniform(-0.1, 0.1, (beta_v.size))
    
    for step in range(num_iterations): 
        
        # Log the current values
        transformed_energies[step] = transformed_energy 
        energies[step] = get_energy( 
                param_alpha=alpha_v, 
                param_beta=beta_v, 
                param_residues=residues, 
                param_residue_positions=residue_positions
                )
        states_alpha[step] = alpha_v
        states_beta[step] = beta_v
        
        # Slightly modify the bond and torsion angle vectors 
        random_i = randint(alpha_v.shape[0])
        random_j = randint(beta_v.shape[0])
        new_alpha_v = np.copy(alpha_v)
        new_beta_v = np.copy(beta_v)
        new_alpha_v[random_i] += random_modifications_alpha[step]
        new_beta_v[random_j] += random_modifications_beta[step]
        
        # Calculate the changes in energy level
        new_transformed_energy = get_transformed_energy(
        param_alpha=new_alpha_v, 
        param_beta=new_beta_v, 
        param_o_min=o_min,
        param_residues=residues, 
        param_residue_positions=residue_positions,
        param_stun=stun
        ) 
        energy_change = new_transformed_energy - transformed_energy
        
        # get the boltzmann factor corresponding to the change
        boltzmann_factor = np.exp(-inv_temps[step]*energy_change)
        
        # accept or reject the new vectors using the metropolis condition
        if random_numbers[step] < boltzmann_factor: 
            alpha_v = new_alpha_v   
            transformed_energy += energy_change
            accepts[step] = 1
        else: 
            rejects[step] = 1
        
        # find a new minimum for stochastic tunneling to improve itself
        if energy_change < 0: 
            o_min = energies[step]
            stun+=0.25

    num_accepts, num_rejects = np.cumsum(accepts, axis=0), np.cumsum(rejects, axis=0)

    annealing_attributes = { 
        'param_alpha': alpha_v,
        'param_beta':  beta_v, 
        'param_energies': energies,
        'param_transformed_energies': transformed_energies,
        'param_stun': stun, 
        'param_states_alpha': states_alpha,  
        'param_o_mins': o_min,
        'param_states_beta': states_beta,
        'param_num_accepts': num_accepts, 
        'param_num_rejects': num_rejects, 
        'param_inv_temps': inv_temps
    }

    return AnnealingOutput(**annealing_attributes)