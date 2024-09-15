import numpy as np 
from dataclasses import dataclass
from numpy.typing import ArrayLike

@dataclass
class AnnealingOutput:
    alpha: ArrayLike 
    beta: ArrayLike 
    energies: ArrayLike
    transformed_energies: ArrayLike
    stun: ArrayLike
    states_alpha: ArrayLike
    min_states_alpha: ArrayLike
    min_states_beta: ArrayLike
    states_beta: ArrayLike
    num_accepts: ArrayLike
    num_rejects: ArrayLike
    inv_temps: ArrayLike

def coefficient(
    i: float, 
    j: float, 
) -> float: 
    
    """

    Return the coefficient determining the strength of interactions between two residues
    If both are 1, then it returns 1 
    If one is 0, it returns 0.5 
    If both are 0, it returns 0.5

    :param i: First residue 
    :param j: Second residue  
    """
    if (i + j == 2): 
        return 1.0
    elif (i+j == 1): 
        return 0.5 
    elif (i+j == 0): 
        return 0.5

# Vectorize the function
coefficient = np.vectorize(coefficient)

def residue_positions(
    size: int,
    alphas: ArrayLike, 
    betas: ArrayLike,
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
    cos_alpha = np.cos(alphas)
    cos_beta = np.cos(betas)
    sin_alpha = np.sin(alphas)
    sin_beta = np.sin(betas)
    positions = np.zeros((size, 3), dtype=float)
    positions[0] = np.array([0,0,0])
    positions[1] = np.array([0,1,0])
    positions[2] = positions[1] + np.array([cos_alpha[0], sin_alpha[0], 0])
    for i in range(size-3): 
        positions[i+3] = positions[i+2]+np.array([cos_alpha[i+1]*cos_beta[i], sin_alpha[i+1]*cos_beta[i], sin_beta[i]]) 
    
    return positions

def energy(
        alpha: ArrayLike, 
        beta: ArrayLike, 
        residues: ArrayLike, 
        residue_positions: ArrayLike,
        k_1: float = 1.0, 
        k_2: float = 0.5
) -> float: 
    
    """ 

    Protein Conformation Energy Function 
    
    :param alpha: Bond angle vector
    :param beta: Torsion angle vector
    :param residues: Array of the residue variables 
    :param residue_positions: Array of the positions of residues
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy
    :returns: float containing the potential energy value
    
    The energy is split into backbone bending energy, torsion energy and energy based on the hydrophobic interactions of residues.
    The third energy is computed using matrices and the required mask. 
    """
    
    backbone_bending_energy = np.dot(np.full((alpha.shape), -k_1), np.cos(alpha))
    torsion_energy = np.dot(np.full((beta.shape), -k_2), np.cos(beta))

    # Mask to select the required elements for the energy sum 
    mask = np.eye((residues.shape, residues.shape), k = np.arange(2, residues.shape))
    
    # Computation of the distance matrix using a vector of residue positions. 
    # An identity matrix of desired length is added to prevent division by zero. This does not affect the final calcluation as diagonal values are not included in the mask. 
    distance_matrix = np.linalg.norm(residue_positions[np.newaxis, :] - residue_positions, axis=-1) + np.eye((residue_positions.shape[0], residue_positions.shape[0])) 
    distance_matrix = distance_matrix**(-6) - distance_matrix**(-12)
    
    return backbone_bending_energy + torsion_energy + 4*distance_matrix*coefficient(residues[np.newaxis, :], residues)

def transformed_energy( 
        alpha: ArrayLike, 
        beta: ArrayLike, 
        min_states_alpha: ArrayLike, 
        min_states_beta: ArrayLike,
        residues: ArrayLike,
        residue_positions: ArrayLike,
        stun: float = 1.0,
        k_1: float = 1.0,
        k_2: float = 0.5
) -> float:
    
    """ 
    
    Transformed Energy Function
    
    :param alpha: Bond angle vector
    :param beta: Torsion angle vector 
    :param states_alpha: Array of the minimum alpha states 
    :param states_beta: Array of the minimum beta states 
    :param residues: Array of the residue variables 
    :param residue_positions: Array of the positions of residues
    :param stun: Controls how much the objective function is flattened 
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy 
    :returns: float containing the transformed energy value
    """
    return 1-np.exp(-stun*energy(alpha, beta, residues, residue_positions, k_1, k_2))

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
    :param k_1: Weight paramter for backbone bending energy. Default = 1.0 
    :param k_2: Weight paramter for torsion energy. Default = 0.5
    :return: AnnealingOutput
    :raises ValueError: if alpha is not of size N-2 '
                        if beta is not of size N-3
    """

    if(init_alpha.shape !=  residues.shape-2 or init_beta.shape != residues.shape-3): 
        raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')