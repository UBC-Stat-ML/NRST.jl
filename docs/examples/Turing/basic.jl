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
model = Lnmodel(randn(30));

# Build an NRST sampler for the model, tune it, sample with it, and show diagnostics
ns  = NRSTSampler(model, verbose = true);
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns,res);
plot(plots..., layout = (3,2), size = (800,1000))

# save cover image #hide
savefig(plots[3], "../covers/basic.svg") #hide
