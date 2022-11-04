using NRST
using Distributions

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
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

# Distribution of the scaled potential function
function get_scaled_V_dist(tm,b)
    s² = sbsq(tm,b)
    s  = sqrt(s²)
    μ  = tm.m*(b*s²-1)/s
    NoncentralChisq(tm.d,tm.d*μ*μ)
end

rng = SplittableRandom(3990)
tm  = MvNormalTM(32,4.,2.)
ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            use_mean = true,
            maxcor   = 1.0,
            γ        = 0.75
);
res   = parallel_run(ns, rng, TE=.0, keep_xs=false);
res.toureff