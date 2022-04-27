# NRST

A Julia package implementing Non-reversible Simulated Tempering. Take a look at the documentation to understand how `NRST` works.

## Building the documentation

The script `build_docs.sh` builds the documentation and then uses [LiveServer.jl](https://github.com/tlienart/LiveServer.jl) to serve the documentation locally.

1. Add execution permissions to build_docs.sh: `$ chmod +x build_docs.sh`.
2. Run the script: `$ ./build_docs.sh -t [nthreads]`, where `nthreads` is the number of threads to use when running the experiments.


