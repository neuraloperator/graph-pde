# Graph based neural operators
This repository contains the code for the two following papers:
- [(GKN) Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)
- [(MGKN) Multipole Graph Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2006.09535)

## Graph Kernel Network (GKN) 
We propose to use graph neural networks for learning the solution operator for partial differential equations. The key innovation in our work is that a single set of network parameters, within a carefully designed network architecture, may be used to describe mappings between infinite-dimensional spaces and between different finite-dimensional approximations of those spaces. 

## Multipole Graph Kernel Network (MGKN)
Inspired by the classical multipole methods, we propose a multi-level graph neural network framework that captures  interaction at all ranges with only linear complexity. Our multi-level formulation is equivalent to recursively adding inducing points to the kernel matrix, unifying GNNs with multi-resolution matrix factorization of the kernel. Experiments confirm our multi-graph network learns discretization-invariant solution operators to PDEs and can be evaluated in linear time.

## Requirements
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)


## Files
The code is in the form of simple scripts. Each script shall be stand-alone and directly runnable.

## Datasets
We provide the Burgers equation and Darcy flow datasets we used in the paper. The data generation can be found in the paper.
The data are given in the form of matlab file. They can be loaded with the scripts provided in utilities.py. 

- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

