using Distributions
using DynamicPPL
using HypothesisTests
using NRST
using SplittableRandoms
using Test

@testset "NRST.jl" begin
    include("Turing_interface.jl")
    include("SliceSampler.jl")
end
