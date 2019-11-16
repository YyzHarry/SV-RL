# Harnessing Structures for Value-Based Planning and Reinforcement Learning

This repository contains the implementation code for paper [Harnessing Structures for Value-Based Planning and Reinforcement Learning](https://arxiv.org/abs/1909.12255).

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


## Experiments
- __[Structured Value-based Planning (SVP)](https://github.com/YyzHarry/SV-RL/tree/master/svp)__
- __[Structured Value-based Reinforcement Learning (SV-RL)](https://github.com/YyzHarry/SV-RL/tree/master/sv_rl)__


## Acknowledgements
We use the implemetation in the [fancyimpute package](https://github.com/iskandr/fancyimpute) for part of our matrix estimation algorithms.
The implementation of SVP is partly based on [this work](https://github.com/haoyio/LowRankMDP).


## Citation
If you find the idea or code useful for your research, please cite [our paper](https://arxiv.org/abs/1909.12255):
```
@article{yang2019harnessing,
  title={Harnessing Structures for Value-Based Planning and Reinforcement Learning},
  author={Yang, Yuzhe and Zhang, Guo and Xu, Zhi and Katabi, Dina},
  journal={arXiv preprint arXiv:1909.12255},
  year={2019},
  url={https://arxiv.org/abs/1909.12255},
}
```
