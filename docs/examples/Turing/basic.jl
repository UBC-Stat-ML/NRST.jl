# ---
# cover: assets/basic_Turing.png
# title: A simple LogNormal-Normal variance model 
# ---

# This demo shows you the basics of performing Bayesian inference on Turing 
# models using NRST.

using Distributions, DynamicPPL, Plots
using NRST

# ## Defining and instantiating a Turing model

# Define a model using the `DynamicPPL.@model` macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations.
model = Lnmodel(randn(30))

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following command will
# instantiate an NRSTSampler and tune it.
ns  = NRSTSampler(model, verbose = true);

# Using the tuned sampler, we run 1024 tours in parallel.
res = parallel_run(ns, ntours = 1024);

# ## Visual diagnostics
plots = diagnostics(ns,res);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))

# save cover image #src
mkpath("assets") #src
savefig(plots[end], "assets/basic_Turing.png") #src

