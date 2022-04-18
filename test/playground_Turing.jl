#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Random, Distributions, DynamicPPL, NRST, Plots, StatsBase
using Interpolations, Roots

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
tune!(samplers)
par_res = run!(samplers, ntours = 1000);
# plot(0:ns.np.N, vec(sum(par_res.visits,dims=2)))
# [get_num_produce(s.np.fns.viout) for s in samplers] # they should be different
# oldTE = par_res.toureff

# tune betas
oldbetas = copy(betas)
tune_betas!(ns,par_res,visualize=true)
maximum(abs,betas-oldbetas) # note that change propagates to betas in main 
NRST.tune_explorers!(ns,nsteps=500)
NRST.initialize_c!(ns,nsteps=8*10*ns.nexpl) # should double in every iteration
