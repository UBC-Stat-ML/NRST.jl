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
# THIS FAILS BECAUSE THERMODYNAMIC INTEGRATION FAILS
@model function HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ TDist(3)                    # can't use improper prior in NRST
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
# ns = NRSTSampler(hmodel,verbose=true);
N = 30;
betas = vcat([0.], 2 .^ range(-200,0,N))
ns = NRSTSampler(hmodel, betas=betas, tune = false);
ns.np.c
nsteps = 16000
aggV = similar(ns.np.c)
trVs = [similar(aggV, nsteps) for _ in 0:N];
NRST.collectVs!(ns, trVs, aggV);
NRST.trapez!(ns.np.c, ns.np.betas, aggV);
R = NRST.est_rej_probs(trVs, ns.np.betas, ns.np.c) # compute average rejection probabilities
Λnorm, Λvalsnorm = NRST.get_lambda(ns.np.betas, R);
Λnorm.(ns.np.betas)

plot(β -> Λnorm(β),0.,1.)
oldbetas = copy(betas)            # store old betas to check convergence
copyto!(ns.np.betas,oldbetas)

NRST.optimize_betas!(betas, R)         # tune using the inverse of Λ(β)
reset_explorers!(ns)              # since betas changed, the cached potentials are stale
return abs.(betas - oldbetas) 

res = parallel_run(ns, ntours=4);
res = parallel_run(ns, ntours=1024);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))
