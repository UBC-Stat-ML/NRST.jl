###############################################################################
# Neal's Funnel
###############################################################################

using NRST
using SplittableRandoms
using Distributions
using Random

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Funnel{TF<:AbstractFloat,TI<:Int} <: NRST.TemperedModel
    β_dist::Normal{TF}
    lenx::TI
end
Funnel(;σ=3., d=20) = Funnel(Normal(zero(σ), σ), d)

# methods for the reference
NRST.Vref(tm::Funnel, x) = -sum(xi -> logpdf(tm.β_dist, xi), x)
function Random.rand!(tm::Funnel{TF}, rng, x) where {TF}
    for i in eachindex(x)
        @inbounds x[i] = rand(rng, tm.β_dist)
    end
    return x
end
function Base.rand(tm::Funnel{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the potential
function NRST.V(tm::Funnel{TF}, x) where {TF}
    α_dist = Normal(zero(TF), exp(first(x)))
    acc    = zero(TF)
    @inbounds for i in 2:tm.lenx
        isinf(acc += logpdf(tm.β_dist, x[i]) - logpdf(α_dist, x[i])) && break
    end
    return acc
end

const tm  = Funnel();
const rng = SplittableRandom(1529)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res=parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);

using Plots
X = collect(hcat(res.xarray[end]...)');
scatter(X[:,1], X[:,5])
savefig("funnel.png")

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

const tm  = HierarchicalModel();
const rng = SplittableRandom(3509)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res=parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);
