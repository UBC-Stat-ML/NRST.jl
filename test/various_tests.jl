using NRST
using StatsBase
using StatsPlots
using Distributions
using LinearAlgebra

const d    = 2 # contour plots only for d==2 
const s0   = 2.
const m    = 4.
const s0sq = s0*s0
sbsq(b)    = 1/(1/s0sq + b)
mu(b)      = b*m*sbsq(b)*ones(d)

# true free energy function == -log(Z(b))
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end

# build an NRST sampler, tune exploration kernels, and do initial tuning of c
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- m)),     # likelihood: N(m1, I)
    x->(0.5sum(abs2,x)/s0sq),     # reference: N(0, s0^2I)
    () -> s0*randn(d),            # reference: N(0, s0^2I)
    collect(range(0,1,length=9)), # betas = uniform grid in [0,1]
    50,                           # nexpl
    true                          # tune c using mean energy
);

# build vector of identical copies of ns for safe parallel computations
samplers = NRST.copy_sampler(ns, nthrds = Threads.nthreads());
copyto!(ns.np.c, F.(ns.np.betas))
res = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());

# tour effectiveness
NRST.full_postprocessing!(res)
all(res.toureff .== [(mean(v)^2) / mean(v.*v) for v in eachcol(res.visits)])

means, vars = NRST.estimate(res, x -> all(x .> m))
plot(ns.np.betas, means, ribbon = 1.96sqrt.(vars/res.ntours))
all(vars .< (4 ./ res.toureff))