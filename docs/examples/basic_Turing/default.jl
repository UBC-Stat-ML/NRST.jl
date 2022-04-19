# ---
# cover: assets/basic_Turing_default.png
# title: A simple LogNormal-Normal variance model 
# ---

# This demo shows you the basics of performing Bayesian inference on Turing 
# models using NRST.

using Distributions, DynamicPPL, Plots
using NRST

# ## Defining and instantiating a Turing model

# Define a model using the DynamicPPL macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a Model by a passing a vector of observations.
lnmodel = Lnmodel(randn(30))

# ## Building, tuning, and running NRST in parallel
# We can now build an NRST sampler using the model. The following commands will
# - instantiate an NRSTSampler
# - create copies for running in parallel
# - tune the samplers
# - run a last round and capture the results
ns = NRSTSampler(lnmodel);
samplers = copy_sampler(ns, nthreads = Threads.nthreads());
tune!(samplers);
par_res = run!(samplers, ntours = 1024);

# ## Visual diagnostics
plots = diagnostics(ns,par_res);
plot(plots..., layout = (3,2), size = (800,1000))

# save cover image #src
mkpath("assets") #src
savefig(plots[end], "assets/basic_Turing_default.png") #src

