using Distributions, DynamicPPL, NRST, Plots, StatsBase

#######################################
# draw a sample from the prior
#######################################

# hierarchical model (Yao et al. 2021, §6.3)
@model function HierarchicalModelPrior(N,J)
    Y  = Matrix{Float64}(undef, N, J) # must explicitly declare for the loop to make sense
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                  # can't use improper prior in NRST
    θ  = Vector{Float64}(undef, J) # must explicitly declare for the loop to make sense
    σ  = sqrt(σ²)
    τ  = sqrt(τ²)
    for j in 1:J
        θ[j] ~ Normal(μ,τ)
        for i in 1:N
            Y[i,j] ~ Normal(θ[j],σ)
        end
    end
    return Y
end
nstudents = 20
nschools  = 8
priormodel = HierarchicalModelPrior(nstudents,nschools)
Ysim = priormodel()

# impose some sanity by requiring 
# - an overall positive effect
# - each observation within [-50,50] range
# - much more disagreement between groups than within
done = false
while !done
    Ysim = priormodel()
    done = minimum(Ysim) > -10 && maximum(Ysim) < 50 &&
        std([mean(y) for y in eachcol(Ysim)]) > 4mean([std(y) for y in eachcol(Ysim)])
end
Ysim # inspect
using DelimitedFiles
writedlm("data/simulated8schools.csv", Ysim, ',')
