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
@model function HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(2.,3.)
    σ² ~ InverseGamma(2.,3.)
    μ  ~ Normal(0.,10.)                  # can't use improper prior in NRST
    θ  = Vector{eltype(Y)}(undef, J)     # must explicitly declare it for the loop to make sense
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
Y = readdlm("data/simulated8schools.csv", ',', Float64)
hmodel = HierarchicalModel(Y);
# N = 30
# Nh = 30 ÷ 2
# exp_grid = 2. .^ range(-min(23, Nh-1), -3, Nh)
# betas = vcat([0.], exp_grid, collect(range(exp_grid[end],1.,N-Nh+1))[2:end])
# plot(range(0,1,N+1),betas)
ns = NRSTSampler(hmodel,verbose=true);
# tune_explorers!(ns,max_rounds=16)
sigmas = [t[1] for t in NRST.params.(ns.explorers)];
plot(sigmas)
ns.np.c
NRST.initialize_c!(ns,nsteps=4000)
NRST.renew!(ns)
res = post_process(NRST.run!(ns,nsteps=10000));
plot(vec(sum(res.visits,dims=2)))

