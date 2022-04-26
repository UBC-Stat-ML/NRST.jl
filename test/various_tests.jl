using Distributions, DynamicPPL, Plots
using NRST

# Define a model using the `DynamicPPL.@model` macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations.
model = Lnmodel(randn(30));
ns  = NRSTSampler(model, verbose = true);


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
ns = NRSTSampler(model, N=100, verbose = true);
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns,res);
plot(plots..., layout = (3,2), size = (800,1000))

N = ns.np.N
nsteps = 32000
aggV = similar(ns.np.c);
trVs = [similar(aggV, nsteps) for _ in 0:N];
NRST.collectVs!(ns, trVs, aggV)
NRST.tune_nexpls!(ns.np.nexpls,trVs,cthrsh = 0.01)
plot(ns.np.nexpls,label="0.01")
NRST.tune_nexpls!(ns.np.nexpls,trVs,cthrsh = 0.05)
plot!(ns.np.nexpls,label="0.05")
NRST.tune_nexpls!(ns.np.nexpls,trVs,cthrsh = 0.1)
plot!(ns.np.nexpls,label="0.1")
NRST.tune_nexpls!(ns.np.nexpls,trVs,cthrsh = 0.2)
plot!(ns.np.nexpls,label="0.2")
extrema(ns.np.nexpls)