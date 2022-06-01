using Distributions, Plots
using Plots.PlotMeasures: px
using ColorSchemes: okabe_ito
using NRST

const d    = 2
const s0   = 2.
const m    = 4.
const s0sq = s0*s0;

# Using these we can write expressions for ``\mu_b``, ``s_b^2``, and ``\mathcal{F}``
sbsq(b) = 1/(1/s0sq + b)
mu(b)   = b*m*sbsq(b)*ones(d)
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end
V(x)      = 0.5sum(abs2,x .- m)
Vref(x)   = 0.5sum(abs2,x)/s0sq
randref() = s0*randn(d);


ns, ts = NRSTSampler(
    V,
    Vref,
    randref,
    N = 3,
    verbose = true,
    do_stage_2 = false
);
copyto!(ns.np.c, F.(ns.np.betas)) # use optimal tuning
res   = NRST.parallel_run(ns, ntours=ts.ntours, keep_xs=false);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/mvNormals/diags.png)


# ## compare sample distribution of scaled 2V/s_b^2 against analytic
function get_scaled_V_dist(b)
    s² = sbsq(b)
    s  = sqrt(s²)
    μ  = m*(b*sbsq(b)-1)/s
    NoncentralChisq(d,d*μ*μ)
end
xpls    = NRST.replicate(ns.xpl, ns.np.betas);
trVs, _ = NRST.collectVs(ns.np, xpls, ts.nsteps);
resser  = NRST.SerialRunResults(NRST.run!(ns, nsteps=2*ns.np.N*ts.ntours));
restur  = NRST.run_tours!(ns, ntours=ts.ntours, keep_xs=false);
parr = []
for (i,trV) in enumerate(trVs)
    β     = ns.np.betas[i]
    sctrV = (2/sbsq(β)) .* trV
    p = plot(
        get_scaled_V_dist(β), label="True", palette=okabe_ito,
        title="β=$(round(β,digits=2))"
    )
    density!(p, sctrV, label="IndExps", linestyle =:dash)
    sctrV = (2/sbsq(β)) .* resser.trVs[i]
    density!(p, sctrV, label="SerialNRST", linestyle =:dash)
    sctrV = (2/sbsq(β)) .* restur.trVs[i]
    density!(p, sctrV, label="TourNRST", linestyle =:dash)
    sctrV = (2/sbsq(β)) .* res.trVs[i]
    density!(p, sctrV, label="pTourNRST", linestyle =:dash)
    push!(parr, p)
end
N  = ns.np.N
nc = min(N+1, ceil(Int,sqrt(N+1)))
nr = ceil(Int, (N+1)/nc)
for i in (N+2):(nc*nr)
    push!(parr, plot(ticks=false, showaxis = false, legend = false))
end
pdists = plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr)
)

#md # ![Bivariate density plots of two neighbors](assets/mvNormals/pdists.png)


# save cover image and diagnostics plots #src
pcover = plot(parr[end-1])
pathnm = "assets/mvNormals" #src
mkpath(pathnm) #src
savefig(pdists, joinpath(pathnm,"diags.png")) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src
