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
V(x)      = 0.5mapreduce(xi -> abs2(xi - m), +, x) # 0 allocs, versus "x .- m" which allocates a temp
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
    N = 4,
    verbose = true,
    tune = false
);
copyto!(ns.np.c, F.(ns.np.betas)) # use optimal tuning
ns.tr
run!(ns,nsteps=8)
ns.tr
