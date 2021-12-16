# NRST

## Building the documentation

### Prepare the docs environment (only for first time building)

From the `docs` folder, start julia and go to Pkg mode typing `]`. Then run
```julia
(@v1.7) pkg> activate .
```
to swith to the `docs` environment. Now run
```julia
(docs) pkg> dev ..
```
to add NRST as dependency for `docs`.

### Compile the documentation

From the `docs` directory, run
```bash
julia -t 4 --project=. make.jl
```
You can change the threads argument `-t` to any number >1 and the examples should run appropriately.