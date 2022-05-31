module IdealIndexProcesses

using Distributions: Exponential
using StaticArrays: MVector
using Statistics: mean, std
using FillArrays: Fill
import ..NRST: renew!, tour!, run_tours!

export BouncyPDMP, BouncyMC, run_tours!, toureff
abstract type Bouncy end
include("BouncyPDMP.jl") # PDMP with reflective boundaries in [0,1]
include("BouncyMC.jl")   # Markov chain on 0:N × {-1,1}

# common methods
toureff(nhs::Vector{<:Int}) = inv(1 + (std(nhs) / mean(nhs))^2)

end