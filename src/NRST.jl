module NRST

using ColorSchemes: seaborn_colorblind, okabe_ito
using DataFrames: DataFrame
using Distributions: Exponential, Normal, Uniform
using DynamicPPL: DynamicPPL
using Interpolations: interpolate, SteffenMonotonicInterpolation, LinearInterpolation
using LogExpFunctions: logsumexp, logistic
using Plots
using Printf
using ProgressMeter: ProgressMeter
using Random: Random
using SmoothingSplines: SmoothingSpline, fit, predict
using StaticArrays: MVector, SVector
using Statistics
using StatsBase: autocor
using StatsPlots: density
using UnicodePlots: UnicodePlots
using UnPack: @unpack

# NRSTSampler.jl
export NRSTSampler,
    run!,
    tune!,
    post_process,
    run_tours!

# ParallelNRST.jl
export parallel_run,
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

# load core files
include("RunResults.jl")
include("log_partition_utils.jl")
include("TemperedModel.jl")
include("ExplorationKernels.jl")
include("NRSTSampler.jl")
include("ParallelNRST.jl")
include("Turing_interface.jl")
include("inference.jl")
include("tuning.jl")

# sub-modules
include("IdealIndexProcesses/IdealIndexProcesses.jl")
using .IdealIndexProcesses

# load remaining files
include("diagnostics.jl")

end
