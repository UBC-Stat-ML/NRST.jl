# ---
# cover: assets/hierarchical_model_htp.png
# title: Hierarchical model (heavy tailed priors)
# ---

# This is almost exactly the same model defined in 
# [Yao et al. (2021, §6.3)](https://arxiv.org/abs/2006.12335), except for the
# fact that -- since we are restricted to proper priors -- we put a Cauchy prior on ``\mu``.
# The data was simulated by us from the same model, and is reminiscent of the classical
# "Eight Schools Problem" posed by [Rubin (1981)](https://www.jstor.org/stable/1164617).

# ## Defining and instantiating a Turing model

using Distributions, DynamicPPL, Plots, DelimitedFiles
using NRST

# Define a model using the `DynamicPPL.@model` macro.
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

# ## Loading the data and instantiating the model
Y = readdlm(
    joinpath(dirname(pathof(NRST)), "..", "data", "simulated8schools.csv"),
     ',', Float64
);
model = HierarchicalModel(Y);

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following commands will
# instantiate an NRSTSampler and tune it.
ns = NRSTSampler(model, N = 30, verbose = true);

# Using the tuned sampler, we run 1024 tours in parallel.
res = parallel_run(ns, ntours = 1024);

# ## Visual diagnostics
plots = diagnostics(ns,res);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))

# save cover image #src
mkpath("assets") #src
savefig(plots[end], "assets/hierarchical_model_htp.png") #src

