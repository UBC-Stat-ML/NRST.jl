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
res   = parallel_run(ns, ntours = ts.ntours);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

# using KernelDensity
# using StatsPlots
using Dierckx

DEF_PAL = NRST.DEF_PAL
nsub_wish = 32768
ngrid = 50
xr = yr = range(-pi,pi,ngrid)
Xnew   = repeat(reshape(xr, 1, :), ngrid, 1)
Ynew   = repeat(yr, 1, ngrid)
parr = []
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