# Protein Structure Prediction

This project aims to visualize protein conformations by optimizing Irbäck's off lattice energy equation. An example of such an implementation can be found [here](https://pdf.sciencedirectassets.com/272830/1-s2.0-S1476927120X00028/1-s2.0-S1476927118307242/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCdkm%2FumvkofuWygOzEy5INM0KZA6KEAPzT2cFNP93qTgIhAIH7TS%2Bmq6im63PpMmP5j3ZUL5w7RMY9zL%2F8s5IopYvyKrwFCPL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgyiUkwc2yX2u6mUj00qkAV1PtDKhA07RE%2FDlDt%2FqA72%2Byo%2BvmBdik8S3kaMdmJjIhZvgrEYJ3FCjostOk3%2FaTHdRCTxbhy3fRiPFBj%2Bm%2BydMSdG2EqapCo2K8SxzucCRo73umhgyBe5IF0E9c%2FXMsf8g9kvEutXIO7BEUHNvq9W2q9L04PUJjySA7HxTqSi0cwjIFWgj9G4H5aBlFeWVJpEDYOjBxVUe%2FCNN7GwmlxUHj1auWhVyAgCCoudjsy2Ga9sG%2B%2BgbW8trmGKtLDKJblK7VoVdO%2BjfcTBgxv35ZJEfHEY1JaSlPj%2B0ov1WhHNJB365lfQkDDwFgnlD7ATy7Wz9dRiwOzsTMTwKfH%2FErQU4r7zwLn6028ECDc71VgTchlFTRCBfaJjD0GXOdRQ61qjbzUgAuvRr4sUJFYn%2B05saU8LUBQhPcmjlp7%2Fs384hHmTiNyu0DrJndMUSNisY1rXB%2B65KicJfsFh3dhHkuVcnyjnuATe1hpcZyY1AbZT61EhXdCeMs8%2FKDA6MP2VJqBbzzw3id%2F3IAcxqljH%2B%2BLwAEKSvN5rY9U%2B3aAVaS7PypPzBfA8QpFdTpZ8JdqF%2FC6Yvf0nJJaOIv5ZRpctiyn%2BjmT%2BgLC69dH8NXnTM%2FSXH%2BNE7DJ9D6qr3znWJJwsTKYZ65yO5XbbKWlSOhaYHS8Vz8GOGZ9%2B1A68hqwer%2FXrQsemGDd71cuq5LIm6iEhmP1R7q0JFjVPBevF8Aq9LcyL3k%2F4%2F1jHTPXoYjn6ITNY9qEInGlQFSRfNs5H0FUT67nri51MK1Y7M9nyYY4gid3wCS1Iavnk1kdnNO9aG6CFnAx47jacVgY03KYkPXgaJwFPGnuEl41TJZb4zlVLCajNfn66pVcBV%2Be9DxLohV73rDCNoJy3BjqwASojDBSOII3tFjRXINyJd87Eid524CUwFzZ8gV8hLFcO64xJ%2FHT9HA9FEiJwAkfnXEg6kJCKMpLXooBG64jKHYoQBlXojvqX6Q%2FEArpsnxf0FWbxTvRTTGaCFEMBxzh58sV0NdotdXD0hT36WHBJw22eUHbBbVfIioykoKGGsyjrzT2d6Lqq6K82LAm0tkzPHhpLe88ISCurLTEBO8I9jNBrSAdd2xCkKOlNN2qfijoP&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240915T175013Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZB6XC2YC%2F20240915%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3094cf2bc6a39dbe51ee45698563db56b802ff26d70cc8ebc1eade111f856a16&hash=5276c1e85a4365e1bcf0e19ce936cf68c09d41ad2aaa8ada6f1bb775bd623add&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1476927118307242&tid=spdf-aa39e806-d0dd-40dd-bac1-957439896b04&sid=45f6fa0f37a97943f61a62d6be040752d8fdgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1b175a03540556505c&rr=8c3a75374cfba933&cc=us).
A hybrid optimization technique known as genetic annealing is employed to reduce computation times. All data from 10/6/2024 onwards was collected using High Performance Computing.
<br/>
The conformation of proteins is generally expressed using bond and torsion angles. These will serve as the independent variables of our annealing problem. The output will be a
tuple of amino acid residue coordinates. The conformation of a given protein is most likely to be the one where energy is minimum. Hence, finding the bond and torsion angles 
for which the energy function is minimum gives the optimal conformation. 
# Simulated Annealing 
Simulated annealing is an optimization technique that searches solution space for the global minimum through random sampling. In this implementation, the annealing will be 
carried out using the metropolis condition and Boltzmann values. 

## Implementation
### The objective function: 
$$f(\alpha, \beta, \chi) = -k_1 \sum_{i=1}^{N-2} \cos \alpha_i - k_2\sum_{i=1}^{N-3} \cos \beta_i + \sum_{i=1}^{N-2}\sum_{j=i+2}^N 4C(\xi_i, \xi_j)(\frac{1}{r^6} - \frac{1}{r^{12}})$$ <br/>
In my code, this is implemented as <br/>  
$$f(\alpha, \beta, \chi) = \langle -k_1, \cos \mathbf{\alpha} \rangle + \langle -k_2, \cos \mathbf{\beta} \rangle + \sum 4\mathbf{C_{ij}}\mathbf{D_{ij}}$$ <br/> 
where $\alpha$ is the bond angle vector, $\beta$ is the torsion angle vector for a given conformation $\chi$ and the third term is the Lennard Jones potential between any two residues. C gives the coefficient depending on the hydrophobicity of any two residues. The matrix- vector representation of this equation allows for usage of NumPy's vectorization improving computation times. 
### Neighbor method 
A random component of either the bond or torsion angle vector is chosen and altered by a small value. The variation is controlled by a heterogeneous degree parameter $\lambda$ as well as the progress of the algorithm.
### Parameters 
The condition to determine the favorability of a neighbor is the Metropolis condition. The starting and ending temperature are `1.0` and `1e-12` respectively. The cooling coefficient $\gamma = 0.99$ and the heterogeneous degree used for choosing new neighbors $\lambda = 3$
The `AnnealingOutput` class serves as a container for the algorithm output. `run.py` utilizes the algorithm results and diagnostic data provided by objects of this class.
## v1.1
This version is primitive and produces inaccurate results. One issue to fix in the next version is the energy difference sometimes causes `OverFlowError`. The annealing schedule must be optimized. 
## v1.2
This version of the algorithm solves the abovementioned `OverFlowError`. The algorithm is now O(`ml`*`n`), where `ml` is the markov chain length and `n` is the number of iterations. For artificial proteins, the markov chain length is set to 50000. For real proteins, it's set to `10000`. The number of iterations depends on the initial and final temperature, as well as the cooling coefficient. 
## v1.3 
The file `src_np.py` utilizing NumPy delivers results within ~0.5 energy units of the values in the research paper for fibonacci artificial proteins of size 13, 21, and 55, although not consistently. Further versions will aim to improve the frequency of accurate modeling.

## v1.4 
A PyTorch version was written for the future to take advantage of GPU acceleration. The NumPy version was tested on the protein 4RXN and yielded an energy value `-172.059076`, close to the optimal value of `-174.612`. The annealing took just over 15 hours.

## v1.5 
Access to high performance computing allows for parallel annealing runs. The annealer was run 10 times independently for artificial proteins [`ml=50000`].
### LAMMPS Dump Files
From now on, conformation data will be stored in LAMMPS (Large Scale Atomic/Molecular Massively Parallel Simulator) DUMP files. This allows conformation visualization in softwares like OVITO. 

# Genetic Annealing 
The implemented hybrid optimization algorithm aims to maximize the efficiency of the simulated annealing algorithm by patching some of its limitations, namely the high time cost. 
The algorithm runs a population of annealers in parallel. The relative fitness is determined after each Markov chain step. Unfit individuals (<0.5) are culled and fit individuals are reproduced to produce daugther annealers with slight variations. The annealer continues until a fixed number of iterations are achieved. The temperature is adaptive and follows the Lam-Delosme cooling schedule. 

## v1.1 
The first working version was implemented in PyTorch. It had faster convergence than serial simulated annealing but always converged to suboptimal local minima. Further versions will be invested in to adjust the cooling schedule and fitness determination.

## v1.2 
A NumPy version was implemented to take advantage of `np.savetxt()` making LAMMP DUMP file creation easier.
### LAMMPS Dump Files
From now on, conformation data will be stored in LAMMPS (Large Scale Atomic/Molecular Massively Parallel Simulator) DUMP files. This allows conformation visualization in softwares like OVITO. 
