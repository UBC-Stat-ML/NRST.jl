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

using Plots
using Plots.PlotMeasures: px
using ColorSchemes: okabe_ito
using NRST
using NRST.ExamplesGallery: MvNormalTM, free_energy, get_scaled_V_dist

#src # TODO: print source code of MvNormal

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - uses the analytic free-energy to set c
# - sample tours in paralle to show diagnostics
rng = SplittableRandom(0x0123456789abcdfe)
tm  = MvNormalTM(32,4.,2.) # d, m, s0
ns, ts = NRSTSampler(
    tm,
    rng,
    N = 12,
    verbose = true,
    do_stage_2 = false
);
copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use optimal tuning
res   = parallel_run(ns, rng, ntours=ts.ntours, keep_xs=false);
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
ntours = 10_000
sbsq(b)= NRSTExp.ExamplesGallery.sbsq(tm, b)
xpls   = NRST.replicate(ns.xpl, ns.np.betas);
trVs   = NRST.collectVs(ns.np, xpls, rng, ceil(Int, sum(ns.np.nexpls)/ns.np.N)*ntours);
resser = NRST.SerialRunResults(NRST.run!(ns, rng, nsteps=2*ns.np.N*ntours));
restur = NRST.run_tours!(ns, rng, ntours=ntours, keep_xs=false);
resPT  = NRST.rows2vov(NRST.run!(NRST.NRPTSampler(ns),rng,10_000).Vs);
parr   = []
for (i,trV) in enumerate(trVs)
    β     = ns.np.betas[i]
    κ     = (2/sbsq(β))    # scaling factor
    sctrV = κ .* trV
    p = plot(
        get_scaled_V_dist(tm,β), label="True", palette=okabe_ito,
        title="β=$(round(β,digits=2))"
    )
    density!(p, sctrV, label="IndExps", linestyle =:dash)
    sctrV = κ .* resPT[i]
    density!(p, sctrV, label="NRPT", linestyle =:dash)
    sctrV = κ .* resser.trVs[i]
    density!(p, sctrV, label="SerialNRST", linestyle =:dash)
    sctrV = κ .* restur.trVs[i]
    density!(p, sctrV, label="TourNRST", linestyle =:dash)
    sctrV = κ .* res.trVs[i]
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
pcover = parr[N+1] #src
pathnm = "assets/mvNormals" #src
mkpath(pathnm) #src
savefig(pdists, joinpath(pathnm,"dists.png")) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src
