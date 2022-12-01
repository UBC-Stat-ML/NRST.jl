module CompetingSamplers

using IrrationalConstants: logtwo
using LogExpFunctions: logsumexp, log1mexp
using Random: AbstractRNG, RandomDevice, randexp
using StaticArrays: MVector
using StatsBase: sample
using UnPack: @unpack
using NRST

include("utils.jl")

include("GeyerThompson1995.jl")
export GT95Sampler

include("SakaiHukushima2016.jl")
export SH16Sampler

end