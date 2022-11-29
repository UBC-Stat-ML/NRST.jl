###############################################################################
# NRSTSampler
###############################################################################

struct NRSTSampler{T,I<:Int,K<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem} <: AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

###############################################################################
# constructors and initialization methods
###############################################################################

struct BadNException{TI<:Int} <: Exception
    Ncur::TI
    Nopt::TI
end
struct NonIntegrableVException <: Exception end

# constructor that also builds an NRSTProblem and does initial tuning
function NRSTSampler(
    tm::TemperedModel,
    rng::AbstractRNG;
    N::Int              = 10,
    nexpl::Int          = 10, 
    use_mean::Bool      = true,
    reject_big_vs::Bool = true,
    log_grid::Bool      = false,
    tune::Bool          = true,
    adapt_N_rounds::Int = 3, 
    kwargs...
    )
    ns      = init_sampler(tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid)
    TE, Λ   = (NaN,NaN)
    adapt_N = 0
    while tune
        try
            TE, Λ = tune!(ns, rng; check_N=(adapt_N<adapt_N_rounds), kwargs...)
            tune  = false
        catch e
            if e isa BadNException
                @warn "N=$(e.Ncur) too " * (e.Nopt > e.Ncur ? "low" : "high") * 
                      ". Setting N=$(e.Nopt) and restarting."
                N = e.Nopt
                ns = init_sampler(
                    tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid
                )
                adapt_N += 1
            elseif e isa NonIntegrableVException
                @warn "V might not be integrable under the reference. " *
                      "Adjusting the adaptation to this fact and restarting."
                log_grid = true
                ns = init_sampler(
                    tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid
                )
            else
                rethrow(e)
            end
        end        
    end
    return ns, TE, Λ
end

function init_sampler(
    tm::TemperedModel,
    rng::AbstractRNG,
    N::Int,
    nexpl::Int, 
    use_mean::Bool,
    reject_big_vs::Bool,
    log_grid::Bool
    )
    betas = init_grid(N)
    x     = initx(rand(tm, rng), rng)                            # draw an initial point
    curV  = Ref(V(tm, x))
    xpl   = get_explorer(tm, x, curV)
    np    = NRSTProblem(                                         # instantiate an NRSTProblem
        tm, N, betas, similar(betas), use_mean, reject_big_vs, 
        log_grid, fill(nexpl, N), fill(params(xpl), N)
    )
    ip    = MVector(zero(N), one(N))
    return NRSTSampler(np, xpl, x, ip, curV)
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    tm = SimpleTemperedModel(V, Vref, randref)
    NRSTSampler(tm,args...;kwargs...)
end

###############################################################################
# sampling methods
###############################################################################

# communication step
function comm_step!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack np,ip,curV = ns
    @unpack N,betas,c = np
    iprop = sum(ip)                             # propose i + eps
    if iprop < 0                                # bounce below
        ip[1] = zero(I)
        ip[2] = one(I)
        return one(K)
    elseif iprop > N                            # bounce above
        ip[1] = N
        ip[2] = -one(I)
        return one(K)
    else
        i    = ip[1]                            # current index
        nlar = get_nlar(betas[i+1],betas[iprop+1],c[i+1],c[iprop+1],curV[])
        acc  = nlar < randexp(rng)              # accept? Note: U<A <=> A>U <=> -log(A) < -log(U) ~ Exp(1) 
        if acc
            ip[1] = iprop                       # move
        else
            ip[2] = -ip[2]                      # flip direction
        end
    end
    return nlar_2_rp(nlar)
end

#######################################
# RegenerativeSampler interface
#######################################

isinatom(ns::NRSTSampler{T,I}) where {T,I} = (ns.ip[1]==zero(I) && ns.ip[2]==-one(I))

# reset state by sampling from the renewal measure
# note: if isinatom(ns), renew! is the same as applying step!
function renew!(ns::NRSTSampler{T,I}, rng::AbstractRNG) where {T,I}
    refreshx!(ns, rng)
    ns.ip[1] = zero(I)
    ns.ip[2] = one(I)
end

# NRST step = comm_step ∘ expl_step => (X,0,-1) is atom
function step!(ns::NRSTSampler, rng::AbstractRNG)
    rp    = comm_step!(ns, rng) # returns rejection probability
    xplap = expl_step!(ns, rng) # returns explorers' acceptance probability
    return rp, xplap
end

# handling last tour step
function save_last_step_tour!(ns::NRSTSampler{T,I,K}, tr; kwargs...) where {T,I,K}
    save_pre_step!(ns, tr; kwargs...)       # store state at atom
    save_post_step!(ns, tr, one(K), K(NaN)) # we know that (-1,-1) would be rejected if attempted so we store this. also, the expl step would not use an explorer; thus the NaN.
end

# multithreading method with automatic determination of required number of tours
const DEFAULT_α      = .95  # probability of the mean of indicators to be inside interval
const DEFAULT_δ      = .10  # half-width of interval
const DEFAULT_TE_min = 5e-4 # truncate TE's below this value
function min_ntours_TE(TE, α=DEFAULT_α, δ=DEFAULT_δ, TE_min=DEFAULT_TE_min)
    ceil(Int, (4/max(TE_min,TE)) * abs2(norminvcdf((1+α)/2) / δ))
end
const DEFAULT_MAX_TOURS = min_ntours_TE(0.)

function parallel_run(
    ns::NRSTSampler, 
    rng::SplittableRandom;
    TE::AbstractFloat = NaN,
    α::AbstractFloat  = DEFAULT_α,
    δ::AbstractFloat  = DEFAULT_δ,
    ntours::Int       = -1,
    kwargs...
    )
    ntours < 0 && (ntours = min_ntours_TE(TE,α,δ))
    parallel_run(ns, rng, ntours; kwargs...)
end
