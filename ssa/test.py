from ga.src_np import get_coefficient 
import numpy as np 
import timeit 
import matplotlib.pyplot as plt
# residues = np.array([1,0,0,1,0,1,0,1],)
# coeff = 0.5*np.kron(residues, residues.T).reshape(residues.shape[0], residues.shape[0])+0.5

# print(0.5*np.einsum("i, j -> ij", residues, residues) +0.5)
# print(get_coefficient(residues))



def wrapper_ein(): 
    np.einsum("ik, jl", A, B).reshape(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

def wrapper_kron(): 
    np.kron(A,B)
eins = np.zeros(100)
krons = np.zeros(100)
for i in range(1,100): 
    A = np.random.randint(0,10, size=(i,i))
    B = np.random.randint(0,10, size=(i,i))
    
    eins[i-1] =  timeit.timeit(wrapper_ein, number=100)
    krons[i-1] =  timeit.timeit(wrapper_kron, number=100)
    print(eins)
plt.plot(np.arange(1, 101), eins, label='EinSum')
plt.plot(np.arange(1, 101), krons, label='Krons')
plt.legend()
plt.show()