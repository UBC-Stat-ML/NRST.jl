# ---
# cover: assets/basic.svg
# title: A simple LogNormal-Normal variance model
# description: Using NRST with a basic Turing model.
# ---

# This demo shows you the basics of performing Bayesian inference on Turing
# models using NRST.

# ## Implementation using NRST

using Distributions, DynamicPPL, Plots
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
ns  = NRSTSampler(model, verbose = true)
plots = diagnostics(ns, parallel_run(ns, ntours = 524_288, keep_xs = false))
hl = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (800,hl*333))

#md # ![Diagnostics plots](assets/basic_diags.svg)

# save diagnostics plot and cover image #src
mkpath("assets") #src
savefig("assets/basic_diags.svg") #src
savefig(plots[3], "assets/basic.svg") #src


