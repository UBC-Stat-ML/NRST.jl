module NRST

using ColorSchemes: seaborn_colorblind
using DataFrames: DataFrame
using Distributions: Exponential, Normal, Uniform
using DynamicPPL: DynamicPPL
using Interpolations: interpolate, SteffenMonotonicInterpolation
using LogExpFunctions: logsumexp
using Plots
using Printf
using ProgressMeter: ProgressMeter
using Random: Random
using Roots: find_zero
using SmoothingSplines: SmoothingSpline, fit, predict
using StaticArrays: MVector
using Statistics
using StatsBase: autocor
using StatsPlots: density
using Suppressor: @suppress_err
using UnicodePlots: UnicodePlots
using UnPack: @unpack

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
abstract type RunResults end
N(res::RunResults) = length(res.xarray)-1 # retrieve max tempering level

# load files
include("log_partition_utils.jl")
include("TemperedModel.jl")
include("ExplorationKernels.jl")
include("NRSTSampler.jl")
include("ParallelNRST.jl")
include("Turing_interface.jl")
include("inference.jl")
include("tuning.jl")
include("diagnostics.jl")

end
