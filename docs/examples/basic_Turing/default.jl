# ---
# cover: assets/basic_Turing_default.png
# title: A simple LogNormal-Normal variance model 
# ---

# This demo shows you the basics of performing Bayesian inference on Turing 
# models using NRST.

using Distributions, DynamicPPL, Plots, StatsBase, StatsPlots
using NRST

# First, let us define the model using the DynamicPPL macro.
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end 

# Now we instantiate a Model by a passing a vector of observations.
lnmodel = Lnmodel(randn(30))

# We can now build an NRST sampler using the model. The following commands will
# - instantiate an NRSTSampler
# - create copies for running in parallel
# - tune the samplers
# - run a last round and capture the results
ns = NRSTSampler(lnmodel);
samplers = copy_sampler(ns, nthreads = 4);
tune!(samplers);
par_res = run!(samplers, ntours = 1024);

# ## Lambda Plot
betas = ns.np.betas;
Λnorm, _ = NRST.get_lambda(par_res, betas);
NRST.plot_lambda(Λnorm,betas,"")

# ## Plot of the log-partition function
lp_df = log_partition(ns, par_res);
plot(
    betas, lp_df[:,1], ribbon = (lp_df[:,1]-lp_df[:,2], lp_df[:,3]-lp_df[:,1]),
    palette=NRST.DEF_PAL, label = "log(Z(β))", legend = :bottomright
)

# ## Distribution of the tour lengths
tourlengths = NRST.tourlengths(par_res);
p = histogram(
    tourlengths, normalize=true, palette = NRST.DEF_PAL,
    xlabel = "Tour length", ylabel = "Density", label = ""
);
N = ns.np.N
vline!(p,
    [2*(N+1)], palette = NRST.DEF_PAL, 
    linewidth = 4, label = "2N+2=$(2*(N+1))"
)

# ## Density plots
colorgrad = cgrad([NRST.DEF_PAL[1], NRST.DEF_PAL[2]], range(0.,1.,N+1));
pdens = density(vcat(par_res.xarray[1]...),color=colorgrad[1],label="")
for i in 2:(N+1)
    density!(pdens, vcat(par_res.xarray[i]...),color=colorgrad[i],label="")
end
plot(pdens)

# save cover image #src
mkpath("assets") #src
savefig(pdens, "assets/basic_Turing_default.png") #src
