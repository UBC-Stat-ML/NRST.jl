# ---
# cover: assets/basic.png
# title: A simple LogNormal-Normal variance model
# description: Using NRST with a basic Turing model.
# ---

# This demo shows you the basics of performing Bayesian inference on Turing
# models using NRST.

# ## Implementation using NRST

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
ns    = NRSTSampler(model, verbose = true)
res   = parallel_run(ns, ntours = 4_096)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(plots..., layout = (hl,2), size = (900,hl*333),left_margin = 30px)

#md # ![Diagnostics plots](assets/basic_diags.png)

# save diagnostics plot and cover image #src
mkpath("assets") #src
savefig(pdiags, "assets/basic_diags.png") #src
savefig(plots[3], "assets/basic.png") #src


