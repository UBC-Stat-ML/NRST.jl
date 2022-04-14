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
betas = pushfirst!(2 .^(range(-8,0,19)), 0.)
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

# # estimate
# at = 0:ns.np.N
# h=(x->x[1])
# infdf = inference(par_res, h=h, at=at)
# show(infdf, allrows=true)

# estimate Λ
rejrat = par_res.rejecs ./ par_res.visits 
averej = 0.5(rejrat[1:(end-1),1] + rejrat[2:end,2]) # average outgoing and incoming rejections
Λvals = pushfirst!(cumsum(averej),0.)
Λvalsnorm = Λvals/Λvals[end]
# plot(betas,Λvalsnorm)

Λnorm = interpolate(betas, Λvalsnorm, SteffenMonotonicInterpolation())
sum(abs, Λnorm.(betas) - Λvalsnorm) < 10eps()

# find betas by inverting Λnorm and u
# initialize
Δ  = 1/ns.np.N # equal steps
targetΛ = 0.
newbetas = similar(betas)
newbetas[1] = minimum(betas)
for i in 2:ns.np.N
    targetΛ += Δ
    b1 = newbetas[i-1]
    b2 = min(1.,betas[findfirst(u -> (u>targetΛ), Λvalsnorm)]) # Λnorm^{-1}(targetΛ) cannot exceed this
    newbetas[i] = find_zero(β -> Λnorm(β)-targetΛ, (b1,b2), atol = 0.01*Δ) # set tolerance |Λnorm(β)-target|< 5%Δ 
end
newbetas[ns.np.N+1] = 1.

# visualize the change
p1 = plot(x->Λnorm(x), 0., 1., label = "Λnorm", legend = :topleft, xlim=(0.,1.), ylim=(0.,1.))
plot!(p1, [0.,0.], [0.,0.], label="βold", color = :green)
for (i,b) in enumerate(betas[2:end])
    y=Λvalsnorm[i+1]
    plot!(p1, [b,b], [0.,y], label="", color = :green) # vertical
    plot!(p1, [0,b], [y,y], label="", color = :green, linestyle = :dot)  # horizontal
end
p2 = plot(x->Λnorm(x), 0., 1., label = "Λnorm", legend = :topleft, xlim=(0.,1.), ylim=(0.,1.))
plot!(p2, [0.,0.], [0.,0.], label="βnew", color = :green)
for (i,b) in enumerate(newbetas[2:end])
    y=(0:Δ:1)[i+1]
    plot!(p2, [b,b], [0.,y], label="", color = :green) # vertical
    plot!(p2, [0,b], [y,y], label="", color = :green, linestyle = :dot)  # horizontal
end
plot(p1,p2,layout=(2,1))
maximum(abs,betas-newbetas)
copyto!(ns.np.betas, newbetas)   # this propagates even to "betas" in main module
NRST.tune_explorers!(ns,nsteps=500)
NRST.initialize_c!(ns,nsteps=8*10*ns.nexpl) # should double in every iteration
par_res = run!(samplers, ntours = 5000);    # this too
plot(0:ns.np.N, vec(sum(par_res.visits,dims=2)))
par_res.toureff .> oldTE

