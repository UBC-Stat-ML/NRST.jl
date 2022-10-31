using NRST
using DynamicPPL, Distributions
using Plots
using Plots.PlotMeasures: px

# lognormal prior, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x .~ Normal(0.,s)
end
model = Lnmodel(randn(30))
tm    = NRST.TuringTemperedModel(model);
rng   = SplittableRandom(4)
ns, TE= NRSTSampler(tm, rng, N=3, use_mean=false);
res   = parallel_run(ns, rng, TE=TE);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

###############################################################################
###############################################################################
using NRST
using DynamicPPL, Distributions
using LinearAlgebra
using Plots
using Plots.PlotMeasures: px
using DelimitedFiles
using Printf
using Random
@model function _HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                  # can't use improper prior in NRST
    θ  ~ MvNormal(fill(μ,J), τ²*I)
    for j in 1:J
        Y[:,j] ~ MvNormal(fill(θ[j], N), σ²*I)
    end
end
# Loading the data and instantiating the model
function HierarchicalModel()
    Y     = readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/simulated8schools.csv", ',', Float64)
    model = _HierarchicalModel(Y)
    return NRST.TuringTemperedModel(model)
end
tm    = HierarchicalModel();
rng   = SplittableRandom(4)
ns    = NRSTSampler(tm, rng, N=10);
res   = parallel_run(ns, rng, ntours = 32_768);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
pcover = scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    markeralpha = min(1., max(0.08, 1000/size(X,2))),
    ylabel="σ: within-group std. dev.", label=""
)