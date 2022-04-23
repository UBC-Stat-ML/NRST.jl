# stepping stone
# Z_N/Z_0 = prod_{i=1}^N Z_i/Z_{i-1}
# <=> log(Z_N/Z_0) = sum_{i=1}^N log(Z_i/Z_{i-1})
# Now,
# Z_i = E^{0}[e^{-beta_i V}] 
# = int pi_0(dx) e^{-beta_i V(x)}
# = int [pi_0(dx) e^{-beta_{i-1} V(x)}] e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} int pi^{i-1}(dx) e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# Hence
# Z_i/Z_{i-1} = E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# ≈ (1/S) sum_{n=1}^{S_{i-1}} e^{-(beta_i-beta_{i-1}) V(x_n)}, x_{1:S_{i-1}} ~ pi^{i-1}
# <=> log(Z_i/Z_{i-1}) ≈ -log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))
#  => log(Z_N/Z_0) = sum_{i=1}^N [-log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))]
# Recipe for the paralell-explorers version
# 1) run collectVs! -> get trVs == V^{i-1}_{1:S_{i-1}}
# 2) compute at each i ∈ (0,N-1): -log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))
# 3) sum all

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
ns = NRSTSampler(model, N = 30, verbose = true);
ns.np.c
res = parallel_run(ns, ntours = 1024);

# ## Visual diagnostics
plots = diagnostics(ns,res);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))
