from src_np import get_conformation as gc 
from src_np import get_energy as ge
import numpy as np 


test_protein = np.random.randint(low=0, high=2, size=13)
test_alphas = np.random.uniform(low=-np.pi, high=np.pi, size=11)
test_betas = np.random.uniform(low=-np.pi, high=np.pi, size=10)
coeffs = 0.5*np.kron(test_protein, test_protein).reshape(test_protein.shape[0], test_protein.shape[0]) + 0.5
test_conformation = gc(13, test_alphas, test_betas)
test_energy = ge(test_alphas, test_betas,test_protein, test_conformation,coeffs)

