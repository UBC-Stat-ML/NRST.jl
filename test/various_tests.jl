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
ns  = NRSTSampler(model, N = 70, verbose = true);
res = parallel_run(ns, ntours = 1024);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))



using Distributions, DynamicPPL, Plots
using NRST

# ## Defining and instantiating a Turing model

# Define a model using the `DynamicPPL.@model` macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations.
model = Lnmodel(randn(30));

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following command will
# instantiate an NRSTSampler and tune it.
ns  = NRSTSampler(model, N=10, verbose = true);

# Using the tuned sampler, we run 1024 tours in parallel.
res = parallel_run(ns, ntours = 1024);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))
ns.np.c
310/44*10

nsteps_expl= 500
round = 18
nsteps = 2^(min(round,8)-1)*nsteps_expl
typeof(nsteps)