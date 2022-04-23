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
ns  = NRSTSampler(model, N=75, verbose = true);
plot(ns.np.betas, ns.np.c)

# stepping stone
α = 0.95
infres = inference(res, h = ns.np.fns.V, at = 0:ns.np.N, α = 1-(1-α)/ns.np.N)
ms     = trapez(ns.np.betas, infres[:,"Mean"])      # integrate the mean
lbs    = trapez(ns.np.betas, infres[:,"C.I. Low"])  # integrate the lower bounds
ubs    = trapez(ns.np.betas, infres[:,"C.I. High"]) # integrate the upper bounds

