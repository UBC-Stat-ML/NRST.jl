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
res = parallel_run(ns, ntours = 524_288, keep_xs = false);
plots = diagnostics(ns, res);
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (800,hl*333))

# ESS versus computational cost
# compares serial v. parallel NRST and against idealizations of the index process
using NRST.IdealIndexProcesses
N = ns.np.N
DEF_PAL = NRST.DEF_PAL
make_log_ticks=NRST.make_log_ticks
Λs = [1.]

cuESS = res.toureff[end]*(1:ntours(res))
lcuESS   = log10.(cuESS)
lcumaxtls = log10.(accumulate(max, tourls))
lcusumtls = log10.(cumsum(tourls))
xlticks = make_log_ticks(lcusumtls) # TODO: if adding more methods, make sure to add them here too
ylticks = make_log_ticks(lcuESS) # TODO: if adding more methods, make sure to add them here too
pcs     = plot(
    xlabel="Computational time",
    ylabel="ESS lower bound", palette = DEF_PAL, 
    legend = :bottomright,
    xticks=(xlticks, ["10^{$e}" for e in xlticks]),
    yticks=(ylticks, ["10^{$e}" for e in ylticks])
)
plot!(pcs, lcusumtls, lcuESS, label = "NRST (ser)")
plot!(pcs, lcumaxtls, lcuESS, label = "NRST (par)")

# add BouncyPDMP
tourls, vNs = run_tours!(BouncyPDMP(Λs[end]), ntours(res))
TE = (sum(vNs) ^ 2) / (ntours(res)*sum(abs2, vNs)) # corresponding ESSlbs[i] = TEs[i]*i, i in 1:ntours 
cuESS = TE*(1:ntours(res))
lcuESS = log10.(cuESS)
scale_tourls = 2(N*tourls .+ 1.)
lcumaxtls = log10.(accumulate(max, scale_tourls))
plot!(pcs, lcumaxtls, lcuESS, label = "Ideal-PDMP")

# add BouncyMC
tourls, vNs = run_tours!(BouncyMC(Λs[end]/N,N), ntours(res))
TE = (sum(vNs) ^ 2) / (ntours(res)*sum(abs2, vNs)) # corresponding ESSlbs[i] = TEs[i]*i, i in 1:ntours 
cuESS = TE*(1:ntours(res))
lcuESS = log10.(cuESS)
lcumaxtls = log10.(accumulate(max,tourls))
plot!(pcs, lcumaxtls, lcuESS, label = "Ideal-MC")

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
ns = NRSTSampler(model, N = 53, verbose = true);
plots = diagnostics(ns, parallel_run(ns, ntours = 524_288, keep_xs = false));
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (800,hl*333))


###############################################################################
# XY model
###############################################################################

using Lattices, Distributions, Plots
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
    N = 57,
    verbose = true
);
plots = diagnostics(ns, parallel_run(ns, ntours = 524_288, keep_xs = false));
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (800,hl*333))
