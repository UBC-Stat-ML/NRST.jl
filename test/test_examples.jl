using Distributions, DynamicPPL, Plots
using Plots.PlotMeasures: px
using NRST

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model = Lnmodel(randn(30))

# This
# - builds an NRST sampler for the model
# - tunes it
# - runs tours in parallel
# - shows diagnostics
ns, ts= NRSTSampler(model, verbose = true)
res   = parallel_run(ns, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)


###############################################################################
###############################################################################
###############################################################################


using Lattices, Distributions, Plots
using Plots.PlotMeasures: px
using NRST

# Define the basics of the model
const S   = 3;
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
    N = 5,
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
DEF_PAL = NRST.DEF_PAL

ngrid     = 50
nsub_wish = ngrid*ngrid*5
parr      = []
for (i,xs) in enumerate(res.xarray)
    # i=1; xs=res.xarray[i]
    β      = ns.np.betas[i]
    nsam   = length(xs)
    nsub   = min(nsub_wish,nsam)
    idx    = sample(1:nsam, nsub, replace=false, ordered=true)
    X      = hcat([x[1:2] for x in xs[idx]]...)
    plev   = scatter(
        X[1,:], X[2,:], markeralpha=1000/nsub, palette=DEF_PAL,
        title="β=$(round(β,digits=2))", label=""
    )
    push!(parr, plev)
end
N  = ns.np.N
nc = min(N+1, ceil(Int,sqrt(N+1)))
nr = ceil(Int, (N+1)/nc)
for i in (N+2):(nc*nr)
    push!(parr, plot())
end 
pcover = plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr), ticks=false, 
    showaxis = false, legend = false, colorbar = false
)