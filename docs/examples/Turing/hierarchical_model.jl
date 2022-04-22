# ---
# cover: assets/hierarchical_model.png
# title: Hierarchical model 
# ---

# This demo shows you the basics of performing Bayesian inference on Turing 
# models using NRST.

using Distributions, DynamicPPL, Plots, DelimitedFiles
using NRST

# This is a simplified version of the hierarchical model in 
# [Yao et al. (2021, §6.3)](https://arxiv.org/abs/2006.12335).
# The data was simulated by us from the true model, and is reminiscent of the classical
# "Eight Schools Problem" posed by [Rubin (1981)](https://www.jstor.org/stable/1164617).

# ## Defining and instantiating a Turing model

# Define a model using the `DynamicPPL.@model` macro.
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

# ## Loading the data and instantiating the model
Y = readdlm(
    joinpath(dirname(pathof(NRST)), "..", "data", "simulated8schools.csv"),
     ',', Float64
);
hmodel = HierarchicalModel(Y);

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following commands will
# - instantiate an NRSTSampler
# - tune the sampler
# - run tours in parallel
# - postprocess the results
ns  = NRSTSampler(hmodel, N = 25, verbose = true);
res = parallel_run(ns, ntours = 1024);

# ## Visual diagnostics
plots = diagnostics(ns,res);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))

# save cover image #src
mkpath("assets") #src
savefig(plots[end], "assets/hierarchical_model.png") #src

