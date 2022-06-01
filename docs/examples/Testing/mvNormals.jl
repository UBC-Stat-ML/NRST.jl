# ---
# cover: assets/mvNormals/cover.png
# title: Multivariate Normals
# description: A model involving multivariate Normals
# ---

# ## The model

# Consider the following statistical model for ``x\in\mathbb{R}^d``
# ```math
# \begin{aligned}
# x   &\sim \mathcal{N}_d(0, s_0^2 I) \\
# y|x &\sim \mathcal{N}_d(x, I).
# \end{aligned}
# ```
# Suppose that we observe ``y = (m,\dots,m) = m1`` for some ``m \in \mathbb{R}``. Then, a potential function ``V`` compatible with this observation is
# ```math
# V(x) := \frac{1}{2}(x - m1)^\top(x - m1).
# ```
# For the prior, we use an unnormalized potential ``V_\text{ref}`` of the form
# ```math
# V_\text{ref}(x) := \frac{1}{2s_0^2}x^\top x.
# ```

# ### Analytical form of the partition function

# For ``b\in[0,1]``, the annealed potential ``V_b`` is given by
# ```math
# \begin{aligned}
# V_b(x) &:= V_0(x) + bV(x)\\
# &= \frac{1}{2}\left[ s_0^{-2}x^\top x  + b(x-m1)^\top (x-m1) \right] \\
# &= \frac{1}{2}\left[ s_0^{-2}x^\top x  + bx^\top x -2bm 1^\top x +bm^21^\top 1 \right] \\
# &= \frac{1}{2}\left[ (s_0^{-2}+b)x^\top x  -2bm 1^\top x  \right] + \frac{1}{2}bdm^2 \\
# &= \frac{1}{2}(s_0^{-2}+b)\left[x^\top x  -2[bm(s_0^{-2}+b)^{-1}1]^\top x  \right] + \frac{1}{2}bdm^2 \\
# &= \frac{1}{2}(s_0^{-2}+b)\left[x^\top x  -2[bm(s_0^{-2}+b)^{-1}1]^\top x  + [b^2m^2(s_0^{-2}+b)^{-2}]1^\top 1 \right] \\
# &\phantom{=} + \frac{1}{2}bdm^2 -  \frac{1}{2}(s_0^{-2}+b)[b^2m^2(s_0^{-2}+b)^{-2}]d \\
# &= \frac{1}{2s_b^2}(x - \mu_b)^\top (x - \mu_b) + \frac{bdm^2}{2}\left[1 - bs_b^2\right],
# \end{aligned}
# ```
# where
# ```math
# \mu_b := bms_b^21, \qquad s_b^2 := (s_0^{-2}+b)^{-1}.
# ```
# Therefore, ``V_b`` corresponds to the energy of a ``\mathcal{N}_d(\mu_b, s_b^2I )`` distribution.
# From this we may infer the normalizing constant ``\mathcal{Z}(b)`` associated to every ``V_b``
# ```math
# \begin{aligned}
# \mathcal{Z}(b) &:= \int \mathrm{d} x \exp\left(-V_b(x)\right) \\
# &= \exp\left(-\frac{bdm^2}{2}\left[1 - bs_b^2\right]\right) \int \mathrm{d} x \exp\left(-\frac{1}{2s_b^2}(x - \mu_b)^\top (x - \mu_b)\right) \\
# &= [2\pi s_b^2]^{d/2} \exp\left(-\frac{bdm^2}{2}\left[1 - bs_b^2\right]\right).
# \end{aligned}
# ```
# It follows that the free energy ``\mathcal{F}`` function is
# ```math
# \begin{aligned}
# \mathcal{F}(b) &= -\log(\mathcal{Z}(b)) \\
# &= -\frac{d}{2}\log(2\pi s_b^2) +\frac{bdm^2}{2}\left[1 - bs_b^2\right] \\
# &= -\frac{d}{2}\left(\log(2\pi s_b^2) -bm^2\left[1 - bs_b^2\right] \right).
# \end{aligned}
# ```
# Recall that, as we know from theory, by using ``c(b) = \mathcal{F}(b)`` we should obtain a uniform distribution over the indices.

# ### Distribution of the potential
 
# We have already seen that ``\pi^{(b)}= \mathcal{N}_d\left(\mu_b, s_b^2I \right)``.
# Now we want to understand the distribution of ``V(x)`` when ``x\sim \pi^{(b)}``.
# Note that
# ```math
# z(x) := \frac{x-m1}{s_b} \sim \mathcal{N}_d\left(\frac{\mu_b-m1}{s_b}, I \right) = \mathcal{N}_d\left(\frac{m}{s_b}(bs_b^2 - 1)1, I \right).
# ```
# It follows that
# ```math
# V(x) = \frac{s_b^2}{2}\|z\|_2^2 \iff \frac{2V(x)}{s_b^2} = \sum_{i=1}^d z_i(x)^2 \sim \chi_d^2\left(d\left[\frac{m(bs_b^2 - 1)}{s_b}\right]^2 \right),
# ```
# where ``\chi_d^2(\lambda)`` is the non-central chi-squared distribution with parameter ``\lambda``.


# ## Implementation using NRST

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

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - uses the analytic free-energy to set c
# - sample tours in paralle to show diagnostics
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

# ## Distribution of the potential

# We compare the sample distribution of ``V(x)`` obtained using various
# strategies against the analytic distribution.
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
parr    = []
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

#md # ![Bivariate density plots of two neighbors](assets/mvNormals/dists.png)

# save cover image and diagnostics plots #src
pcover = parr[end-1] #src
pathnm = "assets/mvNormals" #src
mkpath(pathnm) #src
savefig(pdists, joinpath(pathnm,"dists.png")) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src
