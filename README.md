# LearningPlumbings

This is the python implementation of the paper ["Graph Neural Networks and 3-Dimensional Topology"](https://arxiv.org/abs/2305.05966) based on PyTorch and PyTorch Geometric.

### Requirements
  - PyTorch 2.0.0
  - PyTorch Geometric >=2.2.0
  - Networkx >= 2.8.8
  
### Usage
The Code includes
  - `SupervisedLearning`, where we check the performance of 3 models `GEN+GAT`, `GCN+GCN`, `GCN+GCN` to decide whether or not two plumbing graphs realize a same (homeomorphic) 3-manifold.
  Please execute `main.py` to run.
  - `ReinforcementLearning`, where we utilize `A3C` algorithm to find a sequence of Neumann moves relating to a pair of equivalent plumbing graphs. For training, please execute `main.py`.
  For a test, one can directly execute `test.py`, by which the trained models are loaded from files `model_a.pth` (actor) and `model_c.pth` (critic).
