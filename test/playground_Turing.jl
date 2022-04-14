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
betas = range(0,1,15) .^(2)
ns=NRSTSampler(
    lnmodel, # use the Turing model to build potentials and prior sampling
    betas,
    50,      # nexpl
    true     # tune c using mean energy
);

# parallel
samplers = copy_sampler(ns, nthreads = 4);
par_res = run!(samplers, ntours = 5000);
# plot(0:ns.np.N, vec(sum(par_res.visits,dims=2)))
# [get_num_produce(s.np.fns.viout) for s in samplers] # they should be different
oldTE = par_res.toureff

# estimate
at = [1, ns.np.N+1]
h=(x->x[1])
inference(par_res, h=h, at=at)

# estimate Λ
using Interpolations, Roots

rejrat = par_res.rejecs ./ par_res.visits 
averej = 0.5(rejrat[1:(end-1),1] + rejrat[2:end,1])
Λvals = pushfirst!(cumsum(averej),0.)
Λvalsnorm = Λvals/Λvals[end]
# plot(betas,Λvalsnorm)

Λnorm = interpolate(betas, Λvalsnorm, SteffenMonotonicInterpolation())
sum(abs, Λnorm.(betas) - Λvalsnorm) < 10eps()

# find betas by inverting Λnorm and u
# initialize
b1 = 0.
Δ  = 1/ns.np.N # equal steps
targetΛ = 0.
newbetas = similar(betas)
newbetas[1] = 0.
for i in 2:ns.np.N
    targetΛ += Δ
    b1 = newbetas[i-1]
    b2 = betas[findfirst(u -> (u>targetΛ), Λvalsnorm)] # Λnorm^{-1}(targetΛ) cannot exceed this
    newbetas[i] = find_zero(β -> Λnorm(β)-targetΛ, (b1,b2), atol = 0.01*Δ) # set tolerance |Λnorm(β)-target|< 5%Δ 
end
newbetas[ns.np.N+1] = 1.
copyto!(ns.np.betas, newbetas)
samplers[2].explorers[1].U.betas # change propagates, cause its just pointers!
NRST.initialize_c!(ns,nsteps=1000)
res = post_process(run!(ns,nsteps=50000));
plot(0:ns.np.N, vec(sum(res.visits,dims=2)))
par_res = run!(samplers, ntours = 5000);
par_res.toureff
