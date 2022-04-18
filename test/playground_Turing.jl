#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Random, Distributions, DynamicPPL, NRST, Plots, StatsBase

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end
lnmodel = Lnmodel(randn(30)) # -.1234
betas = collect(range(0,1,20)) #pushfirst!(2 .^(range(-8,0,19)), 0.)
ns=NRSTSampler(
    lnmodel, # use the Turing model to build potentials and prior sampling
    betas,
    50,      # nexpl
    true     # tune c using mean energy
);

# parallel
samplers = copy_sampler(ns, nthreads = 4);
tune!(samplers);
par_res = run!(samplers, ntours = 1024);
# Λnorm, _ = NRST.get_lambda(par_res, betas);
# NRST.plot_lambda(Λnorm,betas,"")

lpdf = log_partition(ns, par_res)
plot(
    betas, lpdf[:,1], ribbon = 0.5*(lpdf[:,3]-lpdf[:,2]),
    palette=NRST.DEFAULT_PALETTE, label = "log(Z(β))", legend = :bottomright
)
