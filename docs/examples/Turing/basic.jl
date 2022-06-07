# ---
# cover: assets/basic/cover.png
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
ns, ts= NRSTSampler(model, N=2, verbose = true)
res   = parallel_run(ns, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/basic/diags.png)

# save cover image and diagnostics plots #src
pcover = plots[:esscost] #src
pathnm = "assets/basic" #src
mkpath(pathnm) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src
