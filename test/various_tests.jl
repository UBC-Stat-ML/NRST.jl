###############################################################################
# Basic model
###############################################################################

using Distributions, DynamicPPL, Plots
using NRST

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model = Lnmodel(randn(30));

# Build an NRST sampler for the model, tune it, sample with it, and show diagnostics
ns = NRSTSampler(model, verbose = true);
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns,res);
plot(plots..., layout = (3,2), size = (800,1000))

###############################################################################
# Hierarchical model
###############################################################################

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
model = HierarchicalModel(Y);

# Build an NRST sampler for the model, tune it, sample with it, and show diagnostics
ns = NRSTSampler(model, N=120, verbose=true, tune=false);
tune!(ns, nsteps_init=256)
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns,res);
plot(plots..., layout = (3,2), size = (800,1000))
extrema(ns.np.nexpls)

###############################################################################
# XY model
###############################################################################


using Lattices, LogExpFunctions, Distributions, Plots
using NRST

# Define the basics of the model
const S   = 8;
const Ssq = S*S;
const sq  = Square(S,S);          # define a square lattice
const βᶜ  = 1.1199;               # critical temp for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003
const J   = 2βᶜ;                  # coupling constant > 1 to force βᶜ < 1 in our parametrization
T(θ)      = logit((θ/pi + 1)/2)   # θ ∈ (-pi,pi) ↦ x ∈ ℝ
Tinv(x)   = pi*(2logistic(x) - 1) # x ∈ ℝ ↦ θ ∈ (-pi,pi)

# Define the potential function
function V(xs::Vector{TF}) where {TF<:AbstractFloat}
    acc = zero(TF)
    θs  = Tinv.(xs)
    for (a, b) in edges(sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end

# Define functions for the transformed reference
const dunif = Uniform(-pi,pi);
randref() = T.(rand(dunif, Ssq))
Vref(x::AbstractFloat) = x + 2log1pexp(-x) # = log1pexp(x) + log1pexp(-x)
Vref(xs::Vector{<:AbstractFloat}) = sum(Vref, xs)

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - sample tours in parallel and uses them to get more accurate estimates of c(β)
# - sample one last time to show diagnostics
ns = NRSTSampler(
    V,
    Vref,
    randref,
    N = 400,
    verbose = true
);
res = parallel_run(ns, ntours = 1024);
NRST.tune_c!(ns, res); # final tuning that uses the more powerful NRST sampling
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns, res);
plot(plots..., layout = (3,2), size = (800,1000))
