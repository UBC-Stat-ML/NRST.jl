using Interpolations

nums = [1, 4, 6, -1, 17, 4, 6]
idxs = findall(i -> i>0,nums)
xs = idxs ./ length(nums)
ys = log.(nums[idxs])
itp = linear_interpolation(xs, ys, extrapolation_bc=Line())

###############################################################################
# HierarchicalModel
###############################################################################

using NRST
using DelimitedFiles
using Distributions
using Random
using SplittableRandoms
const log2π = log(2pi)

#######################################
# pure julia version
# >4 times faster than Turing
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct HierarchicalModel{TF<:AbstractFloat,TI<:Int} <: NRST.TemperedModel
    τ²_prior::InverseGamma{TF}
    σ²_prior::InverseGamma{TF}
    Y::Matrix{TF}
    N::TI
    J::TI
    lenx::TI
end
function HierarchicalModel()
    Y = hm_load_data()
    HierarchicalModel(InverseGamma(.1,.1), InverseGamma(.1,.1), Y, size(Y)..., 11)
end
function hm_load_data()
    readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/simulated8schools.csv", ',')
end
function invtrans(::HierarchicalModel{TF}, x::AbstractVector{TF}) where {TF}
    (τ²=exp(x[1]), σ²=exp(x[2]), μ=x[3], θ = @view x[4:end])
end

# methods for the prior
function NRST.Vref(tm::HierarchicalModel{TF}, x) where {TF}
    τ², σ², μ, θ = invtrans(tm, x)
    acc  = zero(TF)
    acc -= logpdf(tm.τ²_prior, τ²) # τ²
    acc -= x[1]                                               # logdetjac τ²
    acc -= logpdf(tm.σ²_prior, σ²) # σ²
    acc -= x[2]                                               # logdetjac σ²
    acc -= logpdf(Cauchy(), μ)                                # μ
    # acc -= logpdf(MvNormal(Fill(μ,tm.J), τ²*I), θ)            # θ
    acc += 0.5(tm.J * (log2π+log(τ²)) + sum(θᵢ -> abs2(θᵢ - μ), θ)/τ²)
    return acc
end
function Random.rand!(tm::HierarchicalModel, rng, x)
    τ²   = rand(rng, tm.τ²_prior)
    τ    = sqrt(τ²)
    x[1] = log(τ²)
    x[2] = log(rand(rng, tm.σ²_prior))
    μ    = rand(rng, Cauchy())
    x[3] = μ
    for i in 4:tm.lenx
        x[i] = rand(rng, Normal(μ, τ))
    end
    return x
end
function Base.rand(tm::HierarchicalModel{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::HierarchicalModel{TF}, x) where {TF}
    _, σ², _, θ = invtrans(tm, x)
    acc = zero(TF)
    for (j, y) in enumerate(eachcol(tm.Y))
        # acc -= logpdf(MvNormal(Fill(θ[j], tm.N), Σ), y)
        acc += 0.5sum(yᵢ -> abs2(yᵢ - θ[j]), y)/σ²
    end
    acc += 0.5 * tm.J * tm.N * (log2π+log(σ²))
    return acc
end

rng = SplittableRandom(999)
tm  = HierarchicalModel()
ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            xpl_smooth_λ=1e-5
);

using Plots
using Plots.PlotMeasures: px
# using SmoothingSplines

# σs = [first(pars) for pars in ns.np.xplpars]
# lσs= log.(σs)
# plot(lσs)
# N = ns.np.N
# xs = range(inv(N),1.,N)
# λ = 1e-5
# spl = fit(SmoothingSpline, xs, lσs, λ);
# plσs = predict(spl)
# pσs = exp.(plσs)
# plot(xs,lσs)
# plot!(xs,plσs)
# plot(xs,σs)
# plot!(xs,pσs)

# ntours = NRST.min_ntours_TE(TE);
res   = parallel_run(ns,rng,TE=TE,keep_xs=false);
plots = diagnostics(ns, res);
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

###############################################################################
# end
###############################################################################

###############################################################################
# mrnatrans
###############################################################################

using NRST
using DelimitedFiles
using Distributions
using Random
using SplittableRandoms

#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct MRNATrans{TF<:AbstractFloat} <: NRST.TemperedModel
    as::Vector{TF}
    bs::Vector{TF}
    bma::Vector{TF} # b-a
    ts::Vector{TF}
    ys::Vector{TF}
end
MRNATrans(as,bs,ts,ys) = MRNATrans(as,bs,bs .- as,ts,ys)
function MRNATrans()
    MRNATrans(
        [-2., -5., -5., -5., -2.],
        [ 1.,  5.,  5.,  5.,  2.],
        mrna_trans_load_data()...
    )
end

function mrna_trans_load_data()
    dta = readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/transfection.csv", ',')
    ts  = Float64.(dta[2:end,1])
    ys  = Float64.(dta[2:end,3])
    return ts, ys
end

# methods for the prior
# if x~U[a,b] and y = f(x) = 10^x = e^{log(10)x}, then
# P(Y<=y) = P(10^X <= y) = P(X <= log10(y)) = F_X(log(y))
# p_Y(y) = d/dy P(Y<=y) = p_X(log(y)) 1/y = [ind{10^a<= y <=10^b}/(b-a)] [1/y]
# which is the reciprocal distribution or logUniform, so it checks out
function NRST.Vref(tm::MRNATrans{TF}, x) where {TF}
    vr = zero(TF)
    for (i,x) in enumerate(x)
        if x < tm.as[i] || x > tm.bs[i] 
            vr = TF(Inf)
            break
        end
    end
    return vr
end
function Random.rand!(tm::MRNATrans, rng, x)
    for (i, a) in enumerate(tm.as)
        x[i] = a + rand(rng) * tm.bma[i]
    end
    return x
end
function Base.rand(tm::MRNATrans{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, length(tm.as)))
end

# method for the likelihood potential
function NRST.V(tm::MRNATrans{TF}, x) where {TF}
    t₀  = 10^x[1]
    km₀ = 10^x[2]
    β   = 10^x[3]
    δ   = 10^x[4]
    σ   = 10^x[5]
    δmβ = δ - β
    acc = zero(TF)
    for (n, t) in enumerate(tm.ts)
        tmt₀ = t - t₀
        μ    = (km₀ / δmβ) * (-expm1(-δmβ * tmt₀)) * exp(-β*tmt₀)
        isfinite(μ) || (μ = TF(1e4))
        acc -= logpdf(Normal(μ, σ), tm.ys[n])
    end
    return acc
end

rng = SplittableRandom(1957)
tm  = MRNATrans()
ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            xpl_smooth_λ=3
            # log_grid = true
            # tune=false
);

using Plots
using Plots.PlotMeasures: px
using SmoothingSplines

σs = [first(pars) for pars in ns.np.xplpars]
lσs= log.(σs)
plot(lσs)
N = ns.np.N
xs = range(inv(N),1.,N)
λ = 1e-8
spl = fit(SmoothingSpline, xs, lσs, λ);
plσs = predict(spl)
plot(lσs)
plot!(plσs)
plot(σs)
plot!(exp.(plσs))

res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

###############################################################################
# end
###############################################################################

###############################################################################
# Titanic
###############################################################################

using NRST
using DelimitedFiles
using Distributions
using LinearAlgebra
using LogExpFunctions
using Random
using SplittableRandoms
const logtwo = log(2.)

# define the HalfCauchy(0,γ) distribution
struct HalfCauchy{TF<:AbstractFloat} <: Distributions.ContinuousUnivariateDistribution
    C::Cauchy{TF}
end
HalfCauchy(γ=1.0) = HalfCauchy(Cauchy(zero(γ), γ))
function Distributions.logpdf(d::HalfCauchy{TF}, x::Real) where {TF}
    insupport(d, x) ? logtwo + logpdf(d.C, x) : TF(-Inf)
end
Base.rand(rng::AbstractRNG, d::HalfCauchy) = abs(rand(rng, d.C))
Base.minimum(::HalfCauchy{TF}) where {TF} = zero(TF)
Base.maximum(::HalfCauchy{TF}) where {TF} = TF(Inf)

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct TitanicHS{TF<:AbstractFloat, TI<:Int} <: NRST.TemperedModel
    X::Matrix{TF}
    y::BitVector
    Xβ::Vector{TF}
    n::TI
    p::TI
    lenx::TI
    H::HalfCauchy{TF}
    T::TDist{TF}
end
function TitanicHS()
    X, y = titanic_load_data()
    n, p = size(X)
    TitanicHS(X, y, similar(X,n), n, p, 2 + 2p, HalfCauchy(), TDist(3))
end

# copy method. keeps everything common except temp storage Xβ. this is needed
# in order to avoid race conditions when sampling in parallel
function Base.copy(tm::TitanicHS)
    TitanicHS(tm.X, tm.y, similar(tm.X, tm.n), tm.n, tm.p, tm.lenx, tm.H, tm.T)
end

function invtrans(tm::TitanicHS{TF}, x::AbstractVector{TF}) where {TF}
    p = tm.p
    ( τ = x[1], α = x[2], λ = view(x, 3:(p+2)), β = view(x, (p+3):tm.lenx) )
end

# methods for the prior
function NRST.Vref(tm::TitanicHS{TF}, x) where {TF}
    τ, α, λ, β = invtrans(tm, x)
    acc  = zero(TF)
    isinf(acc -= logpdf(tm.H, τ)) && return acc
    isinf(acc -= logpdf(tm.T, α)) && return acc
    for (i, λᵢ) in enumerate(λ)
        isinf(acc -= logpdf(tm.H, λᵢ)) && return acc
        isinf(acc -= logpdf(Normal(zero(TF), λᵢ*τ), β[i])) && return acc
    end
    return acc
end
function Random.rand!(tm::TitanicHS{TF}, rng, x) where {TF}
    τ    = rand(rng, tm.H)
    x[1] = τ
    x[2] = rand(rng, tm.T)
    p    = tm.p
    for i in 3:(p+2)
        λᵢ     = rand(rng, tm.H)
        x[i]   = λᵢ
        x[i+p] = rand(rng, Normal(zero(TF), λᵢ*τ))
    end
    return x
end
function Base.rand(tm::TitanicHS{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::TitanicHS{TF}, x) where {TF}
    _, α, _, β = invtrans(tm, x)
    mul!(tm.Xβ, tm.X, β)
    acc = zero(TF)
    for (i, yᵢ) in enumerate(tm.y)
        @inbounds ℓ = α + tm.Xβ[i]             # 2.5 times faster with @inbounds
        acc += yᵢ ? log1pexp(-ℓ) : log1pexp(ℓ) # log1pexp is never Inf if ℓ isn't, so no need for a check here
    end
    return acc
end

function titanic_load_data()
    dta = readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/titanic.csv", ',', skipstart=1)
    y   = dta[:,1] .> 0
    X   = dta[:,2:end]
    return X, y
end

rng = SplittableRandom(1957)
tm  = TitanicHS()
ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            xpl_smooth_λ=3,
            max_rounds=8
);

using Plots

lσs = [log(first(pars)) for pars in ns.np.xplpars]
plot(lσs)
plot(ns.np.nexpls)
###############################################################################
# end
###############################################################################

###############################################################################
# MvNormal
###############################################################################

using NRST
using Distributions
using Random

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct MvNormalTM{TI<:Int,TF<:AbstractFloat} <: NRST.TemperedModel
    d::TI
    m::TF
    s0::TF
    s0sq::TF
end
MvNormalTM(d,m,s0) = MvNormalTM(d,m,s0,s0*s0)
NRST.V(tm::MvNormalTM, x) = 0.5sum(xi -> abs2(xi - tm.m), x)  # 0 allocs, versus "x .- m" which allocates a temp
NRST.Vref(tm::MvNormalTM, x) = 0.5sum(abs2,x)/tm.s0sq
Random.rand!(tm::MvNormalTM, rng, x) = map!(_ -> tm.s0*randn(rng), x, x)
Base.rand(tm::MvNormalTM, rng) = tm.s0 * randn(rng, tm.d)

# Write methods for the analytical expressions for ``\mu_b``, 
# ``s_b^2``, and ``\mathcal{F}``
sbsq(tm,b) = inv(inv(tm.s0sq) + b)
pars(tm,b) = (s²=sbsq(tm,b);μ = b*tm.m*s²; return (μ, s²))
mu(tm,b)   = first(pars(tm,b)) 
function free_energy(tm::MvNormalTM,b::Real)
    m   = tm.m
    ssq = sbsq(tm, b)
    -0.5*tm.d*( log(2pi) + log(ssq) - b*m*m*(1-b*ssq) )
end
free_energy(tm::MvNormalTM, bs::AbstractVector) = map(b->free_energy(tm,b), bs)

# get the marginal of pibeta
function get_pibeta_mar(tm,b)
    μ, s² = pars(tm,b)
    Normal(μ, sqrt(s²))
end

# Distribution of the scaled potential function
function get_scaled_V_dist(tm,b)
    s² = sbsq(tm,b)
    s  = sqrt(s²)
    μ  = tm.m*(b*s²-1)/s
    NoncentralChisq(tm.d,tm.d*μ*μ)
end

# do special tuning with exact free_energy
rng = SplittableRandom(8371)
tm = MvNormalTM(32,4.,2.)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    reject_big_vs=false,
    γ = 2.0,
    maxcor=0.9
)
copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy

using Plots
using Plots.PlotMeasures: px
using ColorSchemes: okabe_ito

res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
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
iidVs  = map((b -> (d=get_pibeta_mar(tm,b); [NRST.V(tm, rand(d,tm.d)) for _ in 1:10000])), ns.np.betas)
# ntours = NRST.get_ntours(res)
# xpls   = NRST.replicate(ns.xpl, ns.np.betas);
# indVs  = NRST.collectVs(ns.np, xpls, rng, ceil(Int, min(1e6,2ntours*mean(ns.np.nexpls))));
# resser = NRST.SerialRunResults(NRST.run!(ns, rng, nsteps=2*ns.np.N*ntours));
# restur = NRST.run_tours!(ns, rng, ntours=ntours, keep_xs=false);
# resPT  = NRST.rows2vov(NRST.run!(NRST.NRPTSampler(ns),rng,2ntours).Vs);
parr   = []
for (i,vs) in enumerate(iidVs)
    # i=1
    # vs = iidVs[i]
    β     = ns.np.betas[i]
    κ     = (2/sbsq(tm,β))    # scaling factor
    p = plot(
        get_scaled_V_dist(tm,β), label="True", palette=okabe_ito,
        title="β=$(round(β,digits=2))"
    )
    sctrV = κ .* vs
    density!(p, sctrV, label="iid", linestyle =:dash)
    # sctrV = κ .* indVs[i]
    # density!(p, sctrV, label="IndExps", linestyle =:dash)
    # sctrV = κ .* resPT[i]
    # density!(p, sctrV, label="NRPT", linestyle =:dash)
    # sctrV = κ .* resser.trVs[i]
    # density!(p, sctrV, label="SerialNRST", linestyle =:dash)
    # sctrV = κ .* restur.trVs[i]
    # density!(p, sctrV, label="TourNRST", linestyle =:dash)
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

