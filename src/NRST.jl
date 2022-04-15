module NRST

using UnPack,Random,Distributions,Printf,Plots,StatsBase,Statistics,StaticArrays,DynamicPPL
using Interpolations: interpolate, SteffenMonotonicInterpolation
using Roots: find_zero
using DataFrames: DataFrame
using Turing: Turing
using ColorSchemes: seaborn_colorblind

const DEFAULT_PALETTE = seaborn_colorblind

# NRSTSampler.jl
export NRSTSampler,
    run!,
    tune!,
    post_process

# ParallelNRST.jl
export copy_sampler,
    parallel_run

# inference.jl
export point_estimate,
    inference

# Turing_interface.jl
export gen_randref,
    gen_Vref,
    gen_V

# NRSTSampler_tuning.jl
export tune_explorers!,
    initialize_c!,
    initialize!,
    tune_c!,
    tune_betas!

# declarations needed here to fix the fact that the "include"s are 
# processed sequentially, so that things appear undefined even tho they exist
# in later "include"s
# NRSTSampler.jl
abstract type Funs end
abstract type RunResults end
N(res::RunResults) = length(res.xarray)-1 # retrieve max tempering level

# load files
include("utils.jl")
include("ExplorationKernels.jl")
include("NRSTSampler.jl")
include("ParallelNRST.jl")
include("Turing_interface.jl")
include("inference.jl")
include("NRSTSampler_tuning.jl")

end
