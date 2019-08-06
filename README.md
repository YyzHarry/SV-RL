# Harnessing Structures for Value-Based Planning and Reinforcement Learning

This repository contains the implementation code for paper [Harnessing Structures for Value-Based Planning and Reinforcement Learning]() (__In submission__).

We harness ....
We propose Structured Value-based Planning (SVP) for classical stochastic control and planning tasks.
We propose Structured Value-based Deep Reinforcement Learning (SV-RL) for xxx.


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
We provide implementations for three classical problems, the Inverted Pendulum, the Double Integrator and the Cart-Pole. For the first two problems, the state space dimension is 2; while for the Cart-Pole problem, the state space dimension is higher, with 4 dimensions.
Note that for different problems, the sizes of state/action space discretization may vary, which can be modified in [`MDP.jl`]().

**Note:** Since we call Python (for the matrix estimation algorithms) from Julia, you will need to install the [`PyCall`](https://github.com/JuliaPy/PyCall.jl) package, and also place the `fancyimpute` folder in your Python `site-packages` directory.
Solutions for common problems (such as import errors) can be found [here](https://github.com/JuliaPy/PyCall.jl).


## Structured Value-based Reinforcement Learning (SV-RL)


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
