module NRST

using UnPack,Random,Distributions,Printf,Plots,StatsBase,Statistics,DynamicPPL
using StaticArrays: MVector
using StatsPlots: density
using Interpolations: interpolate, SteffenMonotonicInterpolation
using Roots: find_zero
using DataFrames: DataFrame
using Turing: Turing
using ColorSchemes: seaborn_colorblind
using LogExpFunctions: logsumexp

# NRSTSampler.jl
export NRSTSampler,
    run!,
    tune!,
    post_process

# ParallelNRST.jl
export copy_sampler,
    parallel_run,
    ntours,
    tourlengths

# inference.jl
export point_estimate,
    inference,
    log_partition

# Turing_interface.jl
export gen_randref,
    gen_Vref,
    gen_V

# tuning.jl
export tune_explorers!,
    initialize_c!,
    initialize!,
    tune_betas!

# diagnostics.jl
export diagnostics

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
include("tuning.jl")
include("diagnostics.jl")

end
