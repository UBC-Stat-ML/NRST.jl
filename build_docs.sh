#!/usr/bin/env bash

# read params
usagestr="usage: $(basename $0) [-t nthreads]"
while getopts 't:' OPTION; do
  case "$OPTION" in
    t)
      nthreads="$OPTARG"
      ;;
    ?)
      echo $usagestr >&2
      exit 1
      ;;
  esac
done
if [ $OPTIND -eq 1 ]; then
  echo $usagestr >&2
  exit 1
fi

# the following puts the correct path to NRST in docs/Manifest.toml
echo "Setting NRST as dependency of docs..."
julia --project=docs/ -e 'using Pkg; Pkg.develop(Pkg.PackageSpec(path="."))'

# compile docs
# note: The `GKSwstype=nul` environment variable is necessary for gifs to compile successfully
# see e.g.: https://github.com/JuliaPlots/Plots.jl/issues/3664#issuecomment-887365869
# and: https://github.com/JuliaPlots/PlotDocs.jl/blob/20c7d27ca8833d68f10cad89fca4fcdb20634b2c/README.md?plain=1#L12
echo "Compiling docs using $nthreads threads (takes a while)..."
GKSwstype=nul julia -t $nthreads --project=docs/ docs/make.jl

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

