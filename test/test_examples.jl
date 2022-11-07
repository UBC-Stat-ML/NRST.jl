###############################################################################
# HModel
###############################################################################

using NRST
using Distributions
using DelimitedFiles
using LinearAlgebra

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
    readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/simulated8schools.csv", ',', Float64)
end
function invtrans(x::AbstractVector{<:AbstractFloat})
    (τ²=exp(x[1]), σ²=exp(x[2]), μ=x[3], θ = @view x[4:end])
end

# methods for the prior
function NRST.Vref(tm::HierarchicalModel{TF,TI}, x) where {TF,TI}
    τ², σ², μ, θ = invtrans(x)
    acc = zero(TF)
    acc -= logpdf(tm.τ²_prior, τ²) # τ²
    acc -= x[1]                                               # logdetjac τ²
    acc -= logpdf(tm.σ²_prior, σ²) # σ²
    acc -= x[2]                                               # logdetjac σ²
    acc -= logpdf(Cauchy(), μ)                                # μ
    acc -= logpdf(MvNormal(fill(μ,tm.J), τ²*I), θ)            # θ
    return acc
end
function Base.rand(tm::HierarchicalModel{TF,TI}, rng) where {TF,TI}
    x    = Vector{TF}(undef, tm.lenx)
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

# method for the likelihood potential
function NRST.V(tm::HierarchicalModel{TF,TI}, x) where {TF,TI}
    _, σ², _, θ = invtrans(x)
    Σ   = σ²*I
    acc = zero(TF)
    for (j, y) in enumerate(eachcol(tm.Y))
        acc -= logpdf(MvNormal(fill(θ[j], tm.N), Σ), y)
    end
    return acc
end

rng = SplittableRandom(4022)
tm  = HierarchicalModel()
# xpl = NRST.MHSampler(
#     tm,
#     [-3.063630981557572, 3.2587676700037096, 0.15624019056154761, -0.15682674961748058, -0.003155326110453785, 0.032602875443850265, 0.29511317232244105, 0.15558824816926103, -0.04829581697127777, 0.12218279814109545, -0.011037940333033847], [-2.4903339750799756, 5.001664219993296, 0.14569035027491203, -1.2567006426334766, -1.7716583432522763, -0.1705603492024396, -1.27763292277157, -0.01423609021459335, -0.04373776544381763, -2.263258554388622, -0.30989975749606546], Base.RefValue{Float64}(1.4536261324729878), Base.RefValue{Float64}(0.0009987336437640403), Base.RefValue{Float64}(5.846180631791068), Base.RefValue{Float64}(604.7992802889573), Base.RefValue{Float64}(6.450214020739927)
# )
# # NRST.step!(xpl, rng)
# tune!(xpl, rng)
# xpl.sigma

ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            use_mean = false,
            maxcor   = 0.8,
            γ        = 1.0
);
res   = parallel_run(ns, rng, TE=TE, keep_xs=false);



# BAD APPROACH:
# function to capture stdout, stderr, and exit code of a cmd
# SHOULD BE REPLACED BY SIMPLY read(command::Cmd, String)
function execute(cmd::Cmd)
    out = Pipe()
    err = Pipe()

    process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
    close(out.in)
    close(err.in)

    (
      stdout = String(read(out)), 
      stderr = String(read(err)),  
      code = process.exitcode
    )
end

cmd = `qstat -fx $jobid`
out,err,status=execute(cmd)
mbused = parse(Float64, match(r"resources_used.mem = (\d+)kb",out)[1])/1024
mbavai = parse(Float64, match(r"Resource_List.mem = (\d+)mb",out)[1])
mbused/mbavai

npgs = open("/proc/$(getpid())/statm") do io
    split(read(io, String))[1]
end

# GB used by the julia process
parse(Float64,npgs)*parse(Float64, execute(`getconf PAGESIZE`)[1])*(1024)^-3

###############################################################################
# end
###############################################################################

###############################################################################
# mvnormal
###############################################################################

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

###############################################################################
# end
###############################################################################