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
ns, ts = NRSTSampler(tm, rng, N=4);
res   = parallel_run(ns, rng, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)
