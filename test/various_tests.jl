using Distributions, DynamicPPL, Plots
using NRST

# ## Defining and instantiating a Turing model

# Define a model using the DynamicPPL macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a Model by a passing a vector of observations.
lnmodel = Lnmodel(randn(30));
ns = NRSTSampler(lnmodel,verbose=true);
res = parallel_run(ns, ntours=4096);
plot(diagnostics(ns,res)..., layout = (3,2), size = (800,1000))
