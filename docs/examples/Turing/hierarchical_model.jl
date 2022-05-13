# ---
# cover: assets/hierarchical_model/cover.png
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
using Plots.PlotMeasures: px
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
ns    = NRSTSampler(model, N = 11, verbose = true)
res   = parallel_run(ns, ntours = 4_096)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/hierarchical_model/diags.png)

# ## Notes on the results
# ### Inspecting within and between-group std. devs.
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
pcover = scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    ylabel="σ: within-group std. dev.", label=""
)

#md # ![Scatter-plot of within and between-group std. devs.](assets/hierarchical_model/cover.png)

# save cover image and diagnostics plots #src
pathnm = "assets/hierarchical_model" #src
mkpath(pathnm) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src

