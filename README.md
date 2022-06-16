# NRST.jl

A Julia package implementing Non-reversible Simulated Tempering. Take a look at the documentation to understand how `NRST` works.

## Installation and building the documentation

The package is still in development, so it is not registered. The best way to test it is to

1. Clone this repo
2. Build the documentation: the script `build_docs.sh` builds the documentation and then uses [LiveServer.jl](https://github.com/tlienart/LiveServer.jl) to serve the documentation locally.
    1. Add execution permissions to build_docs.sh: `$ chmod +x build_docs.sh`.
    2. Run the script: `$ ./build_docs.sh -t [nthreads]`, where `nthreads` is the number of threads to use when running the experiments.


