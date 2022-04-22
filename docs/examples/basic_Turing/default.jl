# ---
# cover: assets/basic_Turing_default.png
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
lnmodel = Lnmodel(randn(30))

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following commands will
# - instantiate an NRSTSampler
# - tune the sampler
# - run tours in parallel
# - postprocess the results
ns  = NRSTSampler(lnmodel, verbose = true);
res = parallel_run(ns, ntours = 4096);

# ## Visual diagnostics
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))

# save cover image #src
mkpath("assets") #src
savefig(plots[end], "assets/basic_Turing_default.png") #src

