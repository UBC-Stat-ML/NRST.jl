# ---
# cover: assets/XY_model/cover.png
# title: XY model
# description: Simulating the classical XY lattice model.
# ---

# The model is the probability distribution of the angles ``\theta_s \in (-\pi,\pi]`` at each
# of the ``s\in\{1,\dots,S^2\}`` locations of a square lattice ``\mathbb{S}`` of side length
# equal to S. The tempered distributions have densities
# ```math
# \pi^{(\beta)}(\theta) = \frac{e^{-\beta V(\theta)}}{Z(\beta)}
# ```
# where
# ```math
# V(\theta) = -J\sum_{(a,b)\in\mathbb{S}} \cos(\theta_a-\theta_b)
# ```
# for ``J>0``. Note that the reference distribution is uniform on ``(-\pi,\pi)^{S^2}``.


# ## Implementation using NRST

using Lattices, Distributions, Plots
using Plots.PlotMeasures: px
using Dierckx
using NRST

# Define the basics of the model
const S   = 4;
const Ssq = S*S;
const sq  = Square(S,S); # define a square lattice
const J   = 2;           # coupling constant to force βᶜ < 1 in our parametrization, since βᶜ = 1.1199 for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003

# Define the potential function
function V(θs::Vector{TF}) where {TF<:AbstractFloat}
    acc = zero(TF)
    for (a, b) in edges(sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end

# Define functions for the reference
const dunif = Uniform(-pi,pi)
randref() = rand(dunif, Ssq)
Vref(θ::AbstractFloat) = -logpdf(dunif, θ)
Vref(θs::Vector{<:AbstractFloat}) = sum(Vref, θs)

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - sample tours in parallel and uses them to get more accurate estimates of c(β)
# - sample one last time to show diagnostics
ns, ts = NRSTSampler(
    V,
    Vref,
    randref,
    N = 12,
    verbose = true
)
res   = parallel_run(ns, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/XY_model/diags.png)

# ## Notes on the results
# ### Bivariate density plots of two neighbors
#
# Note that for ``\theta_a,\theta_b \in [-\pi,\pi]``,
# ```math
# \cos(\theta_a-\theta_b) = 0 \iff \theta_a - \theta_b = 2k\pi
# ```
# for ``k\in\{-1,0,1\}``. The plots below show that as ``\beta`` increases,
# the samples concentrate at either of three loci, each described by a different
# value of ``k``. Indeed, the diagonal corresponds to ``k=0``, while the
# off-diagonal loci have ``|k|=1``. In the ideal physical model,
# ``\theta_s \in (-\pi,\pi]``, so the non-coherent states have 0 prior probability.
# In floating-point arithmetic, however, the distinction between open and closed
# is impossible.
nsub_wish = 32768
ngrid     = 50
xr = yr = range(-pi,pi,ngrid)
Xnew      = repeat(reshape(xr, 1, :), ngrid, 1)
Ynew      = repeat(yr, 1, ngrid)
parr      = []
for (i,xs) in enumerate(res.xarray)
    # i=1; xs=res.xarray[i]
    β      = ns.np.betas[i]
    nsam   = length(xs)
    nsub   = min(nsub_wish,nsam)
    idx    = sample(1:nsam, nsub, replace=false, ordered=true)
    X      = hcat([x[1:2] for x in xs[idx]]...)
    probs  = exp.(ns.np.c[i] .- β*res.trVs[i][idx])
    pmin,pmax = extrema(probs)
    cprobs = (pmax-pmin<eps()) ? probs : (probs.-pmin)./(pmax-pmin)
    spline = Spline2D(X[1,:], X[2,:], cprobs, s=nsub)
    Z      = map(spline, Xnew, Ynew)
    plev   = scatter(
        X[1,:], X[2,:], markeralpha=1000/nsub, palette=DEF_PAL,
        title="β=$(round(β,digits=2))", label=""
    )
    i > 1 && contour!(plev, xr, yr, Z) # needed because contour is not working with constant Z    
    push!(parr, plev)
end
N  = ns.np.N
nc = min(N+1, ceil(Int,sqrt(N+1)))
nr = floor(Int, (N+1)/nc)
for i in (N+2):(nc*nr)
    push!(parr, plot())
end 
pcover = plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr), ticks=false, 
    showaxis = false, legend = false, colorbar = false
)

#md # ![Bivariate density plots of two neighbors](assets/XY_model/cover.png)


# save cover image and diagnostics plots #src
pathnm = "assets/XY_model" #src
mkpath(pathnm) #src
savefig(pcover, joinpath(pathnm,"cover.png")) #src
savefig(pdiags, joinpath(pathnm,"diags.png")) #src
for (nm,p) in zip(keys(plots), plots) #src
    savefig(p, joinpath(pathnm, String(nm) * ".png")) #src
end #src
