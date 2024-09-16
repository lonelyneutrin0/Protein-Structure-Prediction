# Protein Structure Prediction

This project aims to replicate the results presented [here](https://pdf.sciencedirectassets.com/272830/1-s2.0-S1476927120X00028/1-s2.0-S1476927118307242/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCdkm%2FumvkofuWygOzEy5INM0KZA6KEAPzT2cFNP93qTgIhAIH7TS%2Bmq6im63PpMmP5j3ZUL5w7RMY9zL%2F8s5IopYvyKrwFCPL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgyiUkwc2yX2u6mUj00qkAV1PtDKhA07RE%2FDlDt%2FqA72%2Byo%2BvmBdik8S3kaMdmJjIhZvgrEYJ3FCjostOk3%2FaTHdRCTxbhy3fRiPFBj%2Bm%2BydMSdG2EqapCo2K8SxzucCRo73umhgyBe5IF0E9c%2FXMsf8g9kvEutXIO7BEUHNvq9W2q9L04PUJjySA7HxTqSi0cwjIFWgj9G4H5aBlFeWVJpEDYOjBxVUe%2FCNN7GwmlxUHj1auWhVyAgCCoudjsy2Ga9sG%2B%2BgbW8trmGKtLDKJblK7VoVdO%2BjfcTBgxv35ZJEfHEY1JaSlPj%2B0ov1WhHNJB365lfQkDDwFgnlD7ATy7Wz9dRiwOzsTMTwKfH%2FErQU4r7zwLn6028ECDc71VgTchlFTRCBfaJjD0GXOdRQ61qjbzUgAuvRr4sUJFYn%2B05saU8LUBQhPcmjlp7%2Fs384hHmTiNyu0DrJndMUSNisY1rXB%2B65KicJfsFh3dhHkuVcnyjnuATe1hpcZyY1AbZT61EhXdCeMs8%2FKDA6MP2VJqBbzzw3id%2F3IAcxqljH%2B%2BLwAEKSvN5rY9U%2B3aAVaS7PypPzBfA8QpFdTpZ8JdqF%2FC6Yvf0nJJaOIv5ZRpctiyn%2BjmT%2BgLC69dH8NXnTM%2FSXH%2BNE7DJ9D6qr3znWJJwsTKYZ65yO5XbbKWlSOhaYHS8Vz8GOGZ9%2B1A68hqwer%2FXrQsemGDd71cuq5LIm6iEhmP1R7q0JFjVPBevF8Aq9LcyL3k%2F4%2F1jHTPXoYjn6ITNY9qEInGlQFSRfNs5H0FUT67nri51MK1Y7M9nyYY4gid3wCS1Iavnk1kdnNO9aG6CFnAx47jacVgY03KYkPXgaJwFPGnuEl41TJZb4zlVLCajNfn66pVcBV%2Be9DxLohV73rDCNoJy3BjqwASojDBSOII3tFjRXINyJd87Eid524CUwFzZ8gV8hLFcO64xJ%2FHT9HA9FEiJwAkfnXEg6kJCKMpLXooBG64jKHYoQBlXojvqX6Q%2FEArpsnxf0FWbxTvRTTGaCFEMBxzh58sV0NdotdXD0hT36WHBJw22eUHbBbVfIioykoKGGsyjrzT2d6Lqq6K82LAm0tkzPHhpLe88ISCurLTEBO8I9jNBrSAdd2xCkKOlNN2qfijoP&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240915T175013Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZB6XC2YC%2F20240915%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3094cf2bc6a39dbe51ee45698563db56b802ff26d70cc8ebc1eade111f856a16&hash=5276c1e85a4365e1bcf0e19ce936cf68c09d41ad2aaa8ada6f1bb775bd623add&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1476927118307242&tid=spdf-aa39e806-d0dd-40dd-bac1-957439896b04&sid=45f6fa0f37a97943f61a62d6be040752d8fdgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1b175a03540556505c&rr=8c3a75374cfba933&cc=us) as practice for simulated annealing. 
Later, machine learning will be implemented for hyperparameter optimization so that the algorithm can determine conformations for a wider range of proteins. 

## Simulated Annealing 
Simulated annealing is an optimization technique that searches solution space for the global minimum through random sampling. In this implementation, the annealing will be 
carried out using the metropolis condition and Boltzmann values. If required, stochastic tunneling will be implemented to flatten local minima and enhance the efficiency of 
solution space exploration. The initial parameters will be identical to the ones in the research paper, however later machine learning algorithms will dynamically update them 
through hyperparameter optimization. 

## Protein Conformation Analysis 
The conformation of proteins is generally expressed using bond and torsion angles. These will serve as the independent variables of our annealing problem. The output will be a
tuple of amino acid residue coordinates. The conformation of a given protein is most likely to be the one where energy is minimum. Hence, finding the bond and torsion angles 
for which the energy function is minimum gives the optimal conformation. 

Details about the recursive definition of residue coordinates and energy values are presented in the research paper. 

# Versions 
## v1.1
This version of the algorithm works and produces similar results to the paper. One issue to fix in the next version is the energy difference sometimes causes `OverFlowError` 
which leads to incorrect results. 

