#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Distributions, DynamicPPL, NRST, Plots, StatsBase

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end
lnmodel = Lnmodel(randn(30)) # -.1234
ns = NRSTSampler(lnmodel);

# parallel
samplers = copy_sampler(ns, nthreads = 4);
tune!(samplers);
par_res = run!(samplers, ntours = 1024);
plots = diagnostics(ns,par_res);
plot(plots..., layout = (3,2), size = (800,1000))
