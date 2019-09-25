# Harnessing Structures for Value-Based Planning and Reinforcement Learning

This repository contains the implementation code for paper [Harnessing Structures for Value-Based Planning and Reinforcement Learning]().

This work proposes a generic framework that allows for exploiting the underlying _low-rank structures_ of the state-action value function (_Q_ function), in both planning and deep reinforcement learning.
We verify empirically the wide existence of low-rank _Q_ functions in the context of control and deep RL tasks.
Specifically, we propose (1) Structured Value-based Planning (__SVP__), for classical stochastic control and planning tasks, and (2) Structured Value-based Deep Reinforcement Learning (__SV-RL__), applicable for any value-based techniques to improve performance on deep RL tasks.


## Installation

### Prerequisites
The current code has been tested on Ubuntu 16.04, for both SVP and SV-RL.

- __SVP:__ The SVP part is mainly implemented in [__Julia__](https://julialang.org/) (and a small part in Python) for several classical stochastic control tasks. We use Julia version of `v0.7.0`, which can be downloaded [here](https://julialang.org/downloads/oldreleases.html).
- __SV-RL:__ We provide a PyTorch implementation of SV-RL for deep reinforcement learning tasks.

**Note:** We test SVP implementation on Julia `v0.7.0`, which is not the latest version (and is unmaintained now). You may choose to use later verion of Julia if needed, but we didn't test on other versions.

### Dependencies for SVP
After installing Julia, just use the package manager within Julia to install the following dependencies:
```julia
using Pkg
Pkg.add("IJulia")
Pkg.add("PGFPlots")
Pkg.add("GridInterpolations")
Pkg.add("PyCall")
Pkg.add("ImageMagick")
```

### Dependencies for SV-RL
You can install the dependencies for SV-RL using
```bash
pip install -r requirements.txt
```


## Structured Value-based Planning (SVP)
For the _Q_ matrix of dimension |S|×|A|, at each value iteration, SVP randomly updates a small portion of the _Q(s,a)_ and employs matrix estimation to reconstruct the remaining elements. We show that stochastic control and planning problems can greatly benefit from such a scheme, where much fewer samples (only sample around __20%__ of _(s,a)_ pairs at each iteration) can achieve almost the same policy as the optimal one.
[[Experimental details]](https://github.com/YyzHarry/SV-RL/tree/master/svp)


## Structured Value-based Reinforcement Learning (SV-RL)
SV-RL is applicable for any value-based deep RL methods such as [DQN](https://www.nature.com/articles/nature14236).
Instead of the full _Q_ matrix, SV-RL naturally focuses on the "sub-matrix", corresponding to the sampled batch of states at the current iteration. For each sampled _Q_ matrix, we apply matrix estimation to represent the learning target in a structured way, which poses a low rank regularization on this "sub-matrix" throughout the training process, and hence eventually the _Q_-network's predictions. If the task possesses a low-rank property, this scheme will give a clear guidance on the learning space during training, after which a better policy can be anticipated.
[[Experimental details]](https://github.com/YyzHarry/SV-RL/tree/master/sv_rl)


## Acknowledgements
We use the implemetation in the [fancyimpute package](https://github.com/iskandr/fancyimpute) for part of our matrix estimation algorithms.


## Citation
If you find the idea or code useful for your research, please cite [our paper](https://arxiv.org/abs/1905.11971):
```
@inproceedings{yang2019menet,
  title={ME-Net: Towards Effective Adversarial Robustness with Matrix Estimation},
  author={Yang, Yuzhe and Zhang, Guo and Katabi, Dina and Xu, Zhi},
  booktitle={Proceedings of the 36th International Conference on Machine Learning, {ICML} 2019},
  year={2019},
  url={https://arxiv.org/abs/1905.11971},
}
```
