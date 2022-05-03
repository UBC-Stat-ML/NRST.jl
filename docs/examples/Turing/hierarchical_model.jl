# ---
# cover: assets/hierarchical_model.svg
# title: Hierarchical model
# description: A random effects model coded in Turing.
# ---

# This is almost exactly the same model defined in
# [Yao et al. (2021, §6.3)](https://arxiv.org/abs/2006.12335)
# (code [here](https://github.com/yao-yl/Multimodal-stacking-code/blob/e698da0ccea048f526356822f423aebbacaf7c2f/code/Parametrization%20and%20zero%20avoiding%20priors/random_effect.stan)),
# except for the fact that---since we are restricted to proper priors---we put
# a Cauchy prior on ``\mu``. and is reminiscent of the classical 
# "Eight Schools Problem" posed by
# [Rubin (1981)](https://www.jstor.org/stable/1164617).
# The data was simulated by us from the same model.

# ## Implementation using NRST

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

# Loading the data and instantiating the model
Y = readdlm(
    joinpath(dirname(pathof(NRST)), "..", "data", "simulated8schools.csv"),
     ',', Float64
);
model = HierarchicalModel(Y)

# This
# - builds an NRST sampler for the model
# - tunes it
# - runs tours in parallel
# - shows diagnostics
ns = NRSTSampler(model, N = 200, verbose = true)
res = parallel_run(ns, ntours = 4096)
plots = diagnostics(ns, res)
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (800,hl*333))

#md # ![Diagnostics plots](assets/hierarchical_model_diags.svg)

# save diagnostics plot and cover image #src
mkpath("assets") #src
savefig("assets/hierarchical_model_diags.svg") #src
savefig(plots[3], "assets/hierarchical_model.svg") #src

