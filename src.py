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

def energy(
        alpha: ArrayLike, 
        beta: ArrayLike, 
        residues: ArrayLike, 
        k_1: float = 1.0, 
        k_2: float = 0.5
) -> float: 
    
    """ 

    Protein Conformation Energy Function 

    :param alpha: Bond angle vector
    :param beta: Torsion angle vector
    :param residues: Array of the residue variables 
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy
    :returns: float containing the potential energy value
    :raises: ValueError if alpha does not have N-2 dimensions
                        if beta does not have N-3 dimensions 
    """

    if(alpha.shape !=  residues.shape-2 or beta.shape != residues.shape-3): 
        raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
    backbone_bending_energy = np.dot(np.full((alpha.shape), -k_1), np.cos(alpha))
    torsion_energy = np.dot(np.full((beta.shape), -k_2), np.cos(beta))
    mask = np.eye((residues.shape, residues.shape), k = np.arange(2, residues.shape))

def transformed_energy( 
        alpha: ArrayLike, 
        beta: ArrayLike, 
        min_states_alpha: ArrayLike, 
        min_states_beta: ArrayLike,
        residues: ArrayLike,
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
    :param stun: Controls how much the objective function is flattened 
    :param k_1: Weight parameter for backbone bending energy 
    :param k_2: Weight parameter for the torsion energy 
    :returns: float containing the transformed energy value
    """
    return 1-np.exp(-stun*energy(alpha, beta, residues, k_1, k_2))

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