using Distributions, DynamicPPL, Plots
using Plots.PlotMeasures: px
using NRST

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model = Lnmodel(randn(30))

# This
# - builds an NRST sampler for the model
# - tunes it
# - runs tours in parallel
# - shows diagnostics
ns, ts= NRSTSampler(model, N=5, verbose = true);
res = NRST.parallel_run(ns, ntours=ts.ntours);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

