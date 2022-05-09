module IdealIndexProcesses

using Distributions: Exponential
using StaticArrays: MVector
using Statistics: mean, std
import ..NRST: renew!, tour!

export BouncyPDMP, BouncyMC, run_tours!, toureff
abstract type Bouncy end
include("BouncyPDMP.jl") # PDMP with reflective boundaries in [0,1]
include("BouncyMC.jl")   # Markov chain on 0:N × {-1,1}

# common methods
toureff(nhs::Vector{<:Int}) = inv(1 + (std(nhs) / mean(nhs))^2)

end