# NRST

A Julia package implementing Non-reversible Simulated Tempering.

Take a look at the documentation to understand how `NRST` works. The best way to learn how to use this package is to go through the growing list of case studies.

## Building the documentation

### Prepare the docs environment (only for first time building)

From the `docs` folder, start julia and go to Pkg mode typing `]`. Type
```julia
(@v1.7) pkg> activate .
```
to switch to the `docs` environment. Now add NRST as dependency
```julia
(docs) pkg> dev ..
```

### Compile the documentation

From the `docs` directory, run
```bash
GKSwstype=nul julia -t 4 --project=. make.jl
```
You can change the threads argument `-t` to any number >1 and the examples should run appropriately. The `GKSwstype=nul` environment variable is necessary for gifs to compile successfully (see [here](https://github.com/JuliaPlots/Plots.jl/issues/3664#issuecomment-887365869) and [here](https://github.com/JuliaPlots/PlotDocs.jl/blob/20c7d27ca8833d68f10cad89fca4fcdb20634b2c/README.md?plain=1#L12)).

Once finished, a `build` folder should appear in the `docs` directory.

