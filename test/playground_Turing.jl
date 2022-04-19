#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Random, Distributions, DynamicPPL, NRST, Plots, StatsBase, LinearAlgebra

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
    betas, lpdf[:,1], ribbon = (lpdf[:,1]-lpdf[:,2], lpdf[:,3]-lpdf[:,1]),
    # xlims = (0.99,1), ylims = (0.8*lpdf[end,2],lpdf[end,3]),
    color=NRST.DEFAULT_PALETTE[1], label = "log(Z(β))", legend = :bottomright
)
tourlengths = NRST.tourlengths(par_res)
p = histogram(
    tourlengths, normalize=true, bar_edges=true, palette = NRST.DEFAULT_PALETTE,
    xlabel = "Tour length", ylabel = "Density", label = ""
)
vline!(
    [2*(ns.np.N+1)], palette = NRST.DEFAULT_PALETTE, 
    linewidth = 4, label = "2N+2=$(2*(ns.np.N+1))"
)

