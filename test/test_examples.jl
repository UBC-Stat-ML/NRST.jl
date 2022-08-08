using NRST
using IrrationalConstants: log2π

struct MvNormalTM{TI<:Int,TF<:AbstractFloat} <: NRST.TemperedModel
    d::TI
    m::TF
    s0::TF
    s0sq::TF
end
MvNormalTM(d,m,s0) = MvNormalTM(d,m,s0,s0*s0)
NRST.V(tm::MvNormalTM, x) = 0.5mapreduce(xi -> abs2(xi - tm.m), +, x) # 0 allocs, versus "x .- m" which allocates a temp
NRST.Vref(tm::MvNormalTM, x) = 0.5sum(abs2,x)/tm.s0sq
Base.rand(tm::MvNormalTM, rng) = tm.s0*randn(rng,tm.d)

# Write methods for the analytical expressions for ``\mu_b``, 
# ``s_b^2``, and ``\mathcal{F}``
sbsq(tm,b) = 1/(1/tm.s0sq + b)
mu(tm,b)   = b*tm.m*sbsq(tm,b)*ones(tm.d)
function free_energy(tm::MvNormalTM,b::Real)
    m   = tm.m
    ssq = sbsq(tm, b)
    -0.5*tm.d*( log2π + log(ssq) - b*m*m*(1-b*ssq) )
end
free_energy(tm::MvNormalTM, bs::AbstractVector{<:Real}) = map(b->free_energy(tm,b), bs)

# instantiate
tm = MvNormalTM(32,4.,2.)
Λ  = 5.32 # best estimate of true barrier        

# do special tuning with exact free_energy
N = 11
rng = SplittableRandom(1)
ns, ts = NRSTSampler(
    tm,
    rng,
    N = N,
    verbose = true,
    do_stage_2 = false,
    maxcor = 0.99
)
copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy

using Interpolations

xpls = NRST.replicate(ns.xpl, ns.np.betas) # create a vector of explorers, fully independent of ns and its own explorer
trVs = NRST.collectVs(ns.np,xpls,rng,32)
R = NRST.est_rej_probs(trVs, ns.np.betas, ns.np.c)
get_lambda(betas::Vector{K}, R)
averej  = (R[1:(end-1),1] + R[2:end,2])/2 # average up and down rejections
Λs      = pushfirst!(cumsum(averej), 0.)
Λsnorm  = Λs/Λs[end]
f_Λnorm = interpolate(ns.np.betas, Λsnorm, SteffenMonotonicInterpolation())

using Plots
f_Λnorm(0.55)

plot(f_Λnorm,0.,1.)
betas=ns.np.betas
K = eltype(betas)
monoroot=NRST.monoroot
# find newbetas by inverting f_Λnorm with a uniform grid on the range
N           = length(betas)-1
Δ           = convert(K,1/N)   # step size of the grid
Λtargets    = zero(K):Δ:one(K)
newbetas    = similar(betas)
newbetas[1] = betas[1]         # technically 0., but is safer this way against rounding errors
for i in 2:N
    targetΛ     = Λtargets[i]
    b1          = newbetas[i-1]
    b2          = betas[findfirst(u -> (u>targetΛ), Λsnorm)] # f_Λnorm^{-1}(targetΛ) cannot exceed this
    newbetas[i] = monoroot(β -> f_Λnorm(β)-targetΛ, b1, b2)
end
newbetas[end] = one(K)
copyto!(betas, newbetas)
return (f_Λnorm, Λsnorm, Λs)