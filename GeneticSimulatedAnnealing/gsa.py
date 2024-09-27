import torch 
from dataclasses import dataclass

# Defining required classes and methods 

# Optimizer Object
@dataclass
class solutionObject: 
    alpha: torch.tensor 
    beta: torch.tensor
    optimal_energy: float
    optimal_conformation: torch.tensor
    temps: torch.tensor
    accepts: torch.tensor 
    rejects: torch.tensor 

    """ 
    :returns optimal_energy: Best energy value
    :returns optimal_conformation: Best conformation
    :returns temps: Temperatures 
    :returns accepts: The number of accepts 
    :returns rejects: The number of rejects 
    :returns boltzmen: A tensor of boltzmann values 
    :returns alpha: Bond angle vector 
    :returns beta: Torsion angle vector
    """

#  Return type for each markov chain step
@dataclass
class annealerOutput: 
    energies: torch.Tensor
    boltzmen: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    accepts: int
    rejects: int
    
    """ 
    Annealer Output Container Class
    :returns energies: A tensor of energy values 
    :returns ratios: A tensor of acceptance ratios 
    :returns boltzmen: A tensor of boltzmann values 
    :returns alpha: Bond angle vector 
    :returns beta: Torsion angle vector
    """

# Annealer
@dataclass 
class Annealer: 
    temp: float
    lam: float
    ml: int
    residues: torch.Tensor
    temps: int
    alpha: torch.Tensor=None
    beta: torch.Tensor=None
    k_1: float = -1.0
    k_2: float = 0.5
    
    def get_coefficient(self) -> torch.tensor:
        """
        
        Return the coefficient determining the strength of interactions between two residues
        If both are 1, then it returns 1 
        If one is 0, it returns 0.5 
        If both are 0, it returns 0.5
        
        :param i: First residue 
        :param j: Second residue 
        
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
    
    def get_energy(self, alpha, beta) -> float: 
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
        
        conformation = self.get_conformation(alpha, beta)
        backbone_bending_energy = -self.k_1*torch.sum(torch.cos(alpha))
        torsion_energy = -self.k_2*torch.sum(torch.cos(beta))
        distance_matrix = torch.linalg.norm(conformation[:, None] - conformation, dim=-1)
        distance_matrix.fill_diagonal_(torch.inf)
        distance_matrix = distance_matrix**(-12) - distance_matrix**(-6)  
        total_energy = backbone_bending_energy + torsion_energy + torch.sum(torch.triu(4*distance_matrix*self.get_coefficient(), diagonal=2))
        return total_energy
    
    def anneal(self) -> annealerOutput: 
        
        """
        :param temp: The temperature of the annealing step 
        :param gamma: Hyperparamter for neighborhood generation 
        :param ml: Markov chain length 
        :params k_1, k_2: Energy hyperparameters  
        :returns: annealerOutput
        :raises ValueError: if angle vectors are of improper dimensions
        """

        # Initializations
        if self.alpha is None: 
            self.alpha = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-2,))
        
        if self.beta is None: 
            self. beta = -torch.pi + 2*torch.pi*torch.rand(size=(self.residues.shape[0]-3,))
        
        if(self.alpha.shape[0] !=  self.residues.shape[0]-2 or self.beta.shape[0] != self.residues.shape[0]-3): 
            raise ValueError(f'The angle vectors are not of the appropriate dimensionality.')
        
        energies=torch.zeros(self.ml,)
        boltzmen = torch.zeros(self.ml,)
        conformations = torch.zeros((self.ml, self.residues.shape[0], 3))
        accepts = torch.zeros(self.ml,)
        rejects = torch.zeros(self.ml,)
        conformation = self.get_conformation(self.alpha, self.beta)
        rannums = torch.rand(self.ml,)
        
        energy = self.get_energy(self.alpha, self.beta)
        
        # Annealing Step 
        
        for i in range(self.ml): 
            # Log Current Values
            energies[i] = energy 
            conformations[i] = conformation
            
            #Update to neighbors 
            random_i = torch.randint(low=0, high=self.alpha.shape[0]+self.beta.shape[0], size=(1,)).item()
            new_alpha_v, new_beta_v = torch.clone(self.alpha), torch.clone(self.beta)
            change = (torch.rand(1).item()-0.5)*torch.rand(1).item()*(1-i/self.temps)**self.lam
            if random_i >= self.alpha.shape[0]: 
                # Prevents out of bound errors
                new_beta_v[random_i - self.alpha.shape[0]] = new_beta_v[random_i - self.alpha.shape[0]] + change if torch.abs(new_beta_v[random_i - self.alpha.shape[0]] + change) < torch.pi else new_beta_v[random_i - self.alpha.shape[0]]-change
            else: 
                # Prevents out of bound errors
                new_alpha_v[random_i] = new_alpha_v[random_i]+change if torch.abs(new_alpha_v[random_i]+change) < torch.pi else new_alpha_v[random_i]-change
            
            new_conformation = self.get_conformation(new_alpha_v, new_beta_v)
            new_energy = self.get_energy(new_alpha_v, new_beta_v)
            energy_change = new_energy-energy
            boltzmen[i] = torch.exp(-energy_change/self.temp)

            if energy_change < 0 or boltzmen[i] > rannums[i]:
                 self.alpha = new_alpha_v
                 self.beta = new_beta_v
                 conformation = new_conformation
                 energy = new_energy
                 accepts[i] = 1
            else:
                rejects[i] = 1
        num_accepts = torch.cumsum(accepts, dim=-1)
        num_rejects = torch.cumsum(rejects, dim=-1)
        returnargs = { 
            'energies': energies, 
            'boltzmen': boltzmen, 
            'alpha': self.alpha, 
            'beta': self.beta, 
            'accepts': num_accepts, 
            'rejects': num_rejects
        }
        return annealerOutput(**returnargs)





        
        
    

# handler: wrapper function for entire algorithm 
def handler( 
        num_annealers: int,
        num_iterations: int, 
        len_markov: int, 
        quality_factor: int,
        callback: callable = None 
)->solutionObject:
    """
    The genetic simulated annealing function
    :param num_annealers: The number of annealers to run in parallel 
    :param num_iterations: The number of temperatures to anneal with 
    :param len_markov: Markov chain length for each annealer (REALLY IMPORTANT)
    :param quality_factor: [1,2] Determines decay rate 
    """


kwargs = { 
    'residues': torch.Tensor([1,0,1,0,1,0]), 
    'temp': 10, 
    'ml': 100, 
    'lam': 3, 
    'temps': 2749
} 
anal = Annealer(**kwargs)
print(anal.anneal())