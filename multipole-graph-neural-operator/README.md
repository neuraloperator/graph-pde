# Multipole Graph Kernel Network (MGKN)

The code for "Multipole Graph Neural Operator for Parametric Partial Differential Equation".
Inspired by the classical multipole methods, 
we propose a novel multi-level graph neural network framework 
that captures  interaction at all ranges with only linear complexity.  
Our multi-level formulation is equivalent 
to recursively adding inducing points to the kernel matrix, 
unifying GNNs with multi-resolution matrix factorization of the kernel.  
Experiments confirm our multi-graph network 
learns discretization-invariant solution operators to PDEs 
and can be evaluated in linear time.

## Requirements
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)


### Files
- multigraph1.py corresonds to section 4.1
- multigraph2.py corresonds to section 4.2
- utilities.py contains helper functions.

### Usage
```
python multigraph1.py
```
```
python multigraph2.py
```
