# NRST

## Building the documentation

From the `docs` directory, run
```
julia -t 4 --project=. make.jl
```
You can change the threads argument `-t` to any number >1 and the examples should run appropriately.