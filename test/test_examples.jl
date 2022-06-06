using Distributions, DynamicPPL, Plots
using Plots.PlotMeasures: px
using NRST

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model = Lnmodel(randn(2))

# This
# - builds an NRST sampler for the model
# - tunes it
# - runs tours in parallel
# - shows diagnostics
ns, ts= NRSTSampler(model, N=5, verbose = true, tune=false)
const tm = ns.np.tm
V(x) = NRST.V(tm, x)
truV(x) = -(x+sum(logpdf.(Normal(0.,exp(x)),model.args.x))) # need to include transform
all([V([x]) ≈ truV(x) for x in randn(100)])
Vref(x) = NRST.Vref(tm, x)
all([Vref([x]) ≈ -logpdf(Normal(),x) for x in randn(100)])
