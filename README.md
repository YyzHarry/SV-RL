# Harnessing Structures for Value-Based Planning and Reinforcement Learning

This repository contains the implementation code for paper [Harnessing Structures for Value-Based Planning and Reinforcement Learning]() (__In submission__).



## Installation

### Prerequisites

#### [__Julia__]()



### Dependencies for SVP

```julia
using Pkg
Pkg.add("IJulia")
Pkg.add("PGFPlots")
Pkg.add("GridInterpolations")
Pkg.add("PyCall")
Pkg.add("ImageMagick")
```

### Dependencies for SV-RL
The current code for SV-RL has been tested on Ubuntu 16.04. The 
You can install the dependencies using
```bash
pip install -r requirements.txt
```

## Structured Value-based Planning (SVP)

We provide implementations for two classical problems, the Inverted Pendulum and the Double Integrator. Note that the only difference for Mountain Car (and Cart-Pole) is the system dynamics, which can be done by modifying [`MDP.jl`]().
