module CompetingSamplers

using IrrationalConstants: logtwo
using Random: AbstractRNG, RandomDevice, randexp
using StaticArrays: MVector
using UnPack: @unpack
using NRST

include("GeyerThompson1995.jl")
export GT95Sampler

end