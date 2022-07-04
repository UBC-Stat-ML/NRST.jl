# NRST.jl

A Julia package implementing Non-reversible Simulated Tempering. Take a look at the documentation to understand how `NRST` works.

## Development installation

1. Clone this repo and `cd` to the folder
2. Run Julia with `$ julia --project` to set the workspace
3. Add unregistered dependencies
```julia
using Pkg
Pkg.add(url="git@github.com:miguelbiron/SplittableRandoms.jl.git")
```

After this, you should be able to load the package
```julia
using NRST
```

### Building the documentation

The quickest way to see this package in action is to build and inspect its documentation.
The script `build_docs.sh` builds the docs and then uses
[LiveServer.jl](https://github.com/tlienart/LiveServer.jl) to serve them locally.
Follow these instructions

1. Add execution permissions: `$ chmod +x build_docs.sh`.
2. Run the script: `$ ./build_docs.sh -t [nthreads]`, where `nthreads` is the number of threads to use when running the experiments.


## Regular installation

1. Start Julia `$ julia`
2. Add unregistered dependencies
```julia
using Pkg
Pkg.add(url="git@github.com:miguelbiron/SplittableRandoms.jl.git")
```
3. Install NRST
```julia
Pkg.add(url="git@github.com:UBC-Stat-ML/NRST.jl.git")
```

After this, you should be able to load the package
```julia
using NRST
```

