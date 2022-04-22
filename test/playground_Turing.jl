#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Distributions, DynamicPPL, NRST, Plots, StatsBase



# #######################################
# # draw a sample from the prior
# #######################################

# # hierarchical model (Yao et al. 2021, §6.3)
# @model function HierarchicalModelPrior(N,J)
#     Y  = Matrix{Float64}(undef, N, J) # must explicitly declare for the loop to make sense
#     τ² ~ InverseGamma(.1,.1)
#     σ² ~ InverseGamma(.1,.1)
#     μ  ~ Normal(0.,40.)                  # can't use improper prior in NRST
#     θ  = Vector{Float64}(undef, J)       # must explicitly declare for the loop to make sense
#     σ  = sqrt(σ²)
#     τ  = sqrt(τ²)
#     for j in 1:J
#         θ[j] ~ Normal(μ,τ)
#         for i in 1:N
#             Y[i,j] ~ Normal(θ[j],σ)
#         end
#     end
#     return Y
# end
# nstudents = 20
# nschools  = 8
# priormodel = HierarchicalModelPrior(nstudents,nschools)
# Ysim = priormodel()

# # impose some sanity by requiring 
# # - an overall positive effect
# # - each observation within [-50,50] range
# # - much more disagreement between groups than within
# done = false
# while !done
#     Ysim = priormodel()
#     done = minimum(Ysim) > -10 && maximum(Ysim) < 50 &&
#         std([mean(y) for y in eachcol(Ysim)]) > 4mean([std(y) for y in eachcol(Ysim)])
# end
# Ysim # inspect
# using DelimitedFiles
# writedlm("data/simulated8schools.csv", Ysim, ',')

#######################################
# build NRST sampler
#######################################

# hierarchical model (Yao et al. 2021, §6.3)
# THIS FAILS because of the integration at beta~0---it explodes---likely due
# to E^{0}[V] = infty. Haven't proved it but the priors are heavy tailed
# DETAILS: 
# - since E^{0}[V] = infty, c becomes (0, ∞, ∞, ..., ∞)
# - but E^{0}[V] = infty is caused by heavy tail, so actually most of the samples
#   from the prior are reasonable. Therefore, the acceptance prob
#       exp(-max(cero, dbs[i]*v - ∞))
#   is 1 for all the samples except the few ∞, so the average is ~1, and so rejection is ~0.
# - Because of this, Λ-tuning does not assign betas close to the origin :( .
@model function HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                    # can't use improper prior in NRST
    θ  = Vector{eltype(Y)}(undef, J) # must explicitly declare it for the loop to make sense
    σ  = sqrt(σ²)
    τ  = sqrt(τ²)
    for j in 1:J
        θ[j] ~ Normal(μ,τ)
        for i in 1:N
            Y[i,j] ~ Normal(θ[j],σ)
        end
    end
end

using DelimitedFiles
Y = readdlm(
    joinpath(dirname(pathof(NRST)), "..", "data", "simulated8schools.csv"),
     ',', Float64
);
hmodel = HierarchicalModel(Y);
ns = NRSTSampler(hmodel, N = 30, use_mean = false, verbose = true);
res = parallel_run(ns, ntours=1024);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))
