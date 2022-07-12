module NRST

using ColorSchemes: seaborn_colorblind, okabe_ito
using DataFrames: DataFrame
using Distributions: Normal, Uniform
using DynamicPPL: DynamicPPL
using Interpolations: interpolate, SteffenMonotonicInterpolation
using LogExpFunctions: logsumexp, logistic
using Plots
using Printf
using ProgressMeter: ProgressMeter
using Random: Random, randexp, AbstractRNG
using SmoothingSplines: SmoothingSpline, fit, predict
using SplittableRandoms: SplittableRandom, split
using StaticArrays: MVector, SVector
using Statistics
using StatsBase: autocor
using StatsPlots: density
using UnicodePlots: UnicodePlots
using UnPack: @unpack

# reexports
export SplittableRandom

# RunResults.jl
export tourlengths

# NRSTSampler.jl
export NRSTSampler,
    run!,
    tune!,
    run_tours!

# ParallelNRST.jl
export parallel_run

# inference.jl
export point_estimate,
    inference,
    log_partition

# tuning.jl
export tune!

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
include("diagnostics.jl")

# sub-modules
include("ExamplesGallery/ExamplesGallery.jl")

end
