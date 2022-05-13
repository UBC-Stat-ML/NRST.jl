###############################################################################
# Basic model
###############################################################################

using Distributions, DynamicPPL, Plots
using Plots.PlotMeasures: px
using NRST

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model = Lnmodel(randn(30));

# Build an NRST sampler for the model, tune it, sample with it, and show diagnostics
ns = NRSTSampler(model, N=3, verbose = true);
res = parallel_run(ns, ntours = 4_096);
plots = diagnostics(ns, res);
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (900,hl*333), left_margin = 30px)

1/(11)*0.1
1/121
###############################################################################
# Hierarchical model
###############################################################################

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
ns    = NRSTSampler(model, N = 11, verbose = true);
res   = parallel_run(ns, ntours = 4_096);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (900,hl*333), left_margin = 30px)

# bivariate plot
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    ylabel="σ: within-group std. dev.", label=""
)

###############################################################################
# XY model
###############################################################################

using Lattices, Distributions, Plots
using Plots.PlotMeasures: px
using NRST

# Define the basics of the model
const S   = 8;
const Ssq = S*S;
const sq  = Square(S,S); # define a square lattice
const J   = 2;           # coupling constant to force βᶜ < 1 in our parametrization, since βᶜ = 1.1199 for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003

# Define the potential function
function V(θs::Vector{TF}) where {TF<:AbstractFloat}
    acc = zero(TF)
    for (a, b) in edges(sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end

# Define functions for the reference
const dunif = Uniform(-pi,pi)
randref() = rand(dunif, Ssq)
Vref(θ::AbstractFloat) = -logpdf(dunif, θ)
Vref(θs::Vector{<:AbstractFloat}) = sum(Vref, θs)

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - sample tours in parallel and uses them to get more accurate estimates of c(β)
# - sample one last time to show diagnostics
ns = NRSTSampler(
    V,
    Vref,
    randref,
    N = 13,
    verbose = true
);
res   = parallel_run(ns, ntours = 4_096);
plots = diagnostics(ns, res);
hl    = ceil(Int, length(plots)/2);
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

# bivariate plot of x[1],x[2] across β
using KernelDensity
using StatsPlots
parr = []
for (i,xs) in enumerate(res.xarray)
    X = hcat([x[1:2] for x in xs]...)
    kdefit = kde(X') 
    push!(
        parr, 
        plot(
            kdefit,# xlabel="x₁", ylabel="x₂", label="",
            title="β=$(round(ns.np.betas[i],digits=2))"
        )
    )
end
N = ns.np.N
nc = min(N+1, 5)
nr = ceil(Int, (N+1)/nc)
for i in (N+2):(nc*nr)
    push!(parr, plot())
end 
plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr), ticks=false, 
    showaxis = false, legend = false, colorbar = false, 
    #bottom_margin = 50px,left_margin = 50px, top_margin = 30px
)
