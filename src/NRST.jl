module NRST

using ColorSchemes: seaborn_colorblind, okabe_ito
using DataFrames: DataFrame
using Distributions: Normal, Uniform
using DynamicPPL: DynamicPPL
using Interpolations: interpolate, FritschButlandMonotonicInterpolation
using LogExpFunctions: logsumexp, logistic
using Plots
using Printf
using ProgressMeter: ProgressMeter
using Random: Random, randexp, AbstractRNG, rand!
using SmoothingSplines: SmoothingSpline, fit, predict
using SplittableRandoms: SplittableRandom, split
using StaticArrays: MVector, SVector
using Statistics
using StatsBase: autocor, winsor
using StatsFuns: norminvcdf
using StatsPlots: density
using UnicodePlots: UnicodePlots
using UnPack: @unpack

# constants
const BIG = 0.01floatmax() # big number but small enough so that BIG*inv(BIG) === 1.0
const LOGSMALL = -750.     # LOGSMALL is negative enough so that exp(LOGSMALL) === 0.

# RunResults.jl
export tourlengths

# MCMCSampler.jl
export run!

# RegenerativeSampler.jl
export run_tours!,
    parallel_run

# NRSTSampler.jl
export NRSTSampler,
    tune!

# NRPTSampler.jl
export NRPTSampler

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
include("utils.jl")
include("TemperedModel.jl")
include("ExplorationKernels.jl")
include("pbs_utils.jl")
include("NRSTProblem.jl")
include("MCMCSampler.jl")
include("RegenerativeSampler.jl")
include("SimulatedTempering.jl")
include("NRSTSampler.jl")
include("NRPTSampler.jl")
include("Turing_interface.jl")
include("inference.jl")
include("tuning.jl")
include("diagnostics.jl")

include("CompetingSamplers/CompetingSamplers.jl")

end
