using Distributions
using DynamicPPL
using HypothesisTests
using NRST
using SplittableRandoms
using Random
using Test

@testset "NRST.jl" begin
    include("Turing_interface.jl")
    include("test_commons.jl")
    include("SliceSampler.jl")
    include("NRPTSampler.jl")
end
