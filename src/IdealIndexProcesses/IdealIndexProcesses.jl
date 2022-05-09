module IdealIndexProcesses

using Distributions: Exponential
using Statistics: mean, std
import ..NRST: renew!, tour!

export Bouncy, run_tours!, toureff
include("Bouncy.jl") # PDMP with reflective boundaries in [0,1]

end