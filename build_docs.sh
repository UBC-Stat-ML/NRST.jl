#!/usr/bin/env bash

# read params
nthreads='0'
build_assets='false'
usagestr="usage: $(basename $0) [-t nthreads]"
while getopts 't:b' flag; do
  case "${flag}" in
    b) build_assets='true' ;;
    t) nthreads="${OPTARG}" ;;
    *) echo $usagestr >&2
       exit 1 ;;
  esac
done
if [ $nthreads -eq 0 ]; then
  echo $usagestr >&2
  exit 1
fi

# grab my fork of DemoCards
myDCdir="$HOME/opt/DemoCards.jl"
if [ -d $myDCdir ] 
then
    echo "Local DemoCards repo exists. Pulling changes..."
    git -C $myDCdir pull
else
    echo "Cloning DemoCards fork..."
    mkdir -p $HOME/opt
    git clone git@github.com:miguelbiron/DemoCards.jl.git $myDCdir
fi

# the following puts the correct path to NRST in docs/Manifest.toml
echo "Setting NRST and DemoCards fork as dependency of docs..."
julia --project=docs/ -e 'using Pkg; Pkg.develop(Pkg.PackageSpec(path=".")); Pkg.develop(Pkg.PackageSpec(path="'$myDCdir'"))'

# compile docs
# note: The `GKSwstype=nul` environment variable is necessary for gifs to compile successfully
# see e.g.: https://github.com/JuliaPlots/Plots.jl/issues/3664#issuecomment-887365869
# and: https://github.com/JuliaPlots/PlotDocs.jl/blob/20c7d27ca8833d68f10cad89fca4fcdb20634b2c/README.md?plain=1#L12
# note: the "include(popfirst!(ARGS))" trick forces proper capturing of ctrl+c (as REPL does)
# see: https://docs.julialang.org/en/v1/base/base/#Core.InterruptException
if [ "$build_assets" = "true" ] 
then
    echo "Building covers and compiling docs using $nthreads threads (takes a while)..."
else
    echo "Compiling docs using $nthreads threads (takes a while)..."
fi

DEMOCARDS_BUILD_ASSETS=$build_assets DOCUMENTER_DEBUG=true JULIA_DEBUG=Documenter GKSwstype=nul julia -t $nthreads --project=docs/ -e "include(popfirst!(ARGS))" docs/make.jl

# serve using LiveServer
jlcode=$(cat << EOF
try
    using LiveServer
catch
    using Pkg
    Pkg.add("LiveServer")
    using LiveServer
end
serve(dir="./docs/build", launch_browser=true)
EOF
)
awk -v var="$jlcode" 'BEGIN {print var}' > tempjlcode.jl
julia tempjlcode.jl
rm tempjlcode.jl

