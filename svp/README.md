# Structured Value-based Planning (SVP)


## Installation
The SVP part is mainly implemented in [__Julia__](https://julialang.org/) (and a small part in Python) for several classical stochastic control tasks. We use Julia version of `v0.7.0`, which can be downloaded [here](https://julialang.org/downloads/oldreleases.html). We test SVP implementation on Julia `v0.7.0`, which is not the latest version (and is unmaintained now). You may choose to use later verion of Julia if needed, but we didn't test on other versions.

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



## Structured Value-based Planning (SVP)
We provide implementations for three classical problems, the Inverted Pendulum, the Double Integrator and the Cart-Pole. For the first two problems, the state space dimension is 2; while for the Cart-Pole problem, the state space dimension is higher, with 4 dimensions.
Note that for different problems, the sizes of state/action space discretization may vary, which can be modified in [`MDP.jl`]().

**Note:** Since we call Python (for the matrix estimation algorithms) from Julia, you will need to install the [`PyCall`](https://github.com/JuliaPy/PyCall.jl) package, and also place the `fancyimpute` folder in your Python `site-packages` directory.
Solutions for common problems (such as import errors) can be found [here](https://github.com/JuliaPy/PyCall.jl).


## Running Examples



## Representative Results
