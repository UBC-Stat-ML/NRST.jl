#!/usr/bin/env bash

# read params
nthreads='0'
usagestr="usage: $(basename $0) [-t nthreads]"
while getopts 't:' flag; do
  case "${flag}" in
    t) nthreads="${OPTARG}" ;;
    *) echo $usagestr >&2
       exit 1 ;;
  esac
done
if [ $nthreads -eq 0 ]; then
  echo $usagestr >&2
  exit 1
fi
echo "Building docs using $nthreads threads."

echo "Setting NRST, my DemoCards fork, and my SplittableRandoms as dependencies of docs..."
sed -i -r '/NRST.+/d' ./docs/Project.toml # remove NRST from the docs Project if exists
jlcmds=$(cat <<-END
using Pkg
Pkg.add(
	[Pkg.PackageSpec(url="git@github.com:miguelbiron/DemoCards.jl.git"),
	 Pkg.PackageSpec(url="git@github.com:miguelbiron/SplittableRandoms.jl.git")]
)
Pkg.develop(Pkg.PackageSpec(path="."))
Pkg.instantiate()
END
)
julia --project=docs/ -e "$jlcmds"

# compile docs
# note: The `GKSwstype=nul` environment variable is necessary for gifs to compile successfully
# see e.g.: https://github.com/JuliaPlots/Plots.jl/issues/3664#issuecomment-887365869
# and: https://github.com/JuliaPlots/PlotDocs.jl/blob/20c7d27ca8833d68f10cad89fca4fcdb20634b2c/README.md?plain=1#L12
# note: the "include(popfirst!(ARGS))" trick forces proper capturing of ctrl+c (as REPL does)
# see: https://docs.julialang.org/en/v1/base/base/#Core.InterruptException
echo "Compiling docs (takes a while)..."
DOCUMENTER_DEBUG=true JULIA_DEBUG=DemoCards GKSwstype=nul julia -t $nthreads --project=docs/ -e "include(popfirst!(ARGS))" docs/make.jl