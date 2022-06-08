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
rng   = SplittableRandom(0x0123456789abcdfe)
ns, ts= NRSTSampler(model, rng, N=2, verbose = true)
res   = parallel_run(ns, rng, ntours = ts.ntours)
res.trvec[end].trV[end] # 82.60585460223331