# NRST.jl

A Julia package implementing Non-reversible Simulated Tempering. Take a look at the documentation to understand how `NRST` works.

## Quick example using the documentation

The quickest way to test this package is to build and inspect its documentation.
The script `build_docs.sh` builds the documentation and then uses
[LiveServer.jl](https://github.com/tlienart/LiveServer.jl) to serve the 
documentation locally. Follow these instructions

1. Clone this repo
2. Add execution permissions: `$ chmod +x build_docs.sh`.
3. Run the script: `$ ./build_docs.sh -t [nthreads]`, where `nthreads` is the number of threads to use when running the experiments.


## Installation

1. Open Julia
2. Add unregistered dependencies
```julia
using Pkg
Pkg.add(url="git@github.com:miguelbiron/SplittableRandoms.jl.git")
```
3. Install NRST
```julia
Pkg.add(url="git@github.com:UBC-Stat-ML/NRST.jl")
```

