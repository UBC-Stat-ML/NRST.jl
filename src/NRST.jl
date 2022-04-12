module NRST

using UnPack,Random,Distributions,Printf,StatsBase,Statistics,StaticArrays,DynamicPPL
using Turing: Turing

# NRSTSampler.jl
export NRSTSampler,
    run!,
    tune!,
    post_process

# ParallelNRST.jl
export copy_sampler,
    parallel_run

# estimation.jl
export estimate

# Turing_interface.jl
export gen_randref,
    gen_Vref,
    gen_V

# declarations needed here to fix the fact that the "include"s are 
# processed sequentially, so that things appear undefined even tho they exist
# abstract type declarations
abstract type Funs end

# load files
include("utils.jl")
include("ExplorationKernels.jl")
include("NRSTSampler.jl")
include("ParallelNRST.jl")
include("NRSTSampler_tuning.jl")
include("estimation.jl")
include("Turing_interface.jl")

end
