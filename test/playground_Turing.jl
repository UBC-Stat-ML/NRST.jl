#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Distributions, DynamicPPL, NRST, Plots, StatsBase
using StatsPlots

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

# plot densities
N = ns.np.N
colorgrad = cgrad([NRST.DEF_PAL[1], NRST.DEF_PAL[2]], range(0.,1.,N+1));
p = density(vcat(par_res.xarray[1]...),color=colorgrad[1],label="")
for i in 2:(N+1)
    density!(p, vcat(par_res.xarray[i]...),color=colorgrad[i],label="")
end
plot(p)

# lambda plot
betas = ns.np.betas;
Λnorm, _ = NRST.get_lambda(par_res, betas);
NRST.plot_lambda(Λnorm,betas,"")

# log partition function plot
lpdf = log_partition(ns, par_res);
plot(
    betas, lpdf[:,1], ribbon = (lpdf[:,1]-lpdf[:,2], lpdf[:,3]-lpdf[:,1]),
    # xlims = (0.99,1), ylims = (0.8*lpdf[end,2],lpdf[end,3]),
    palette=NRST.DEF_PAL, label = "log(Z(β))", legend = :bottomright
)
tourlengths = NRST.tourlengths(par_res);
histogram(
    tourlengths, normalize=true, palette = NRST.DEF_PAL,
    xlabel = "Tour length", ylabel = "Density", label = ""
)
vline!(
    [2*(ns.np.N+1)], palette = NRST.DEF_PAL, 
    linewidth = 4, label = "2N+2=$(2*(ns.np.N+1))"
)

