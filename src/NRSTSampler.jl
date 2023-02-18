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

# outer constructor
function NRSTSampler(
    tm::TemperedModel,
    rng::AbstractRNG,
    ::Type{TXpl}        = SliceSampler;
    N::Int              = 30,
    log_grid::Bool      = false,
    tune::Bool          = true,
    adapt_N_rounds::Int = 3,
    kwargs...
    ) where {TXpl <: ExplorationKernel}
    ns      = init_sampler(tm, rng, TXpl; N=N, log_grid=log_grid, kwargs...)
    stats   = ()
    adapt_N = 0
    while tune
        try
            stats = tune!(ns, rng; check_N=(adapt_N<adapt_N_rounds), kwargs...)
            tune  = false
        catch e
            if e isa BadNException
                @warn "N=$(e.Ncur) too " * (e.Nopt > e.Ncur ? "low" : "high") * 
                      ". Setting N=$(e.Nopt) and restarting."
                N = e.Nopt
                ns = init_sampler(tm, rng, TXpl; N=N, log_grid=log_grid, kwargs...)
                adapt_N += 1
            elseif e isa NonIntegrableVException
                @warn "V might not be integrable under the reference. " *
                      "Adjusting the adaptation to this fact and restarting."
                log_grid = true
                ns = init_sampler(tm, rng, TXpl; N=N, log_grid=log_grid, kwargs...)
            else
                rethrow(e)
            end
        end        
    end
    return (ns, stats...)
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V::Function, Vref::Function, randref::Function, args...;kwargs...)
    tm = SimpleTemperedModel(V, Vref, randref)
    NRSTSampler(tm,args...;kwargs...)
end

# initialize a sampler
function init_sampler(
    tm::TemperedModel,
    rng::AbstractRNG,
    ::Type{TXpl};
    N::Int,
    log_grid::Bool,
    use_mean::Bool      = true,
    reject_big_vs::Bool = true,
    kwargs...
    ) where {TXpl <: ExplorationKernel}
    betas = init_grid(N)
    x     = initx(rand(tm, rng), rng)                            # draw an initial point
    curV  = Ref(V(tm, x))
    xpl   = TXpl(tm, x, one(eltype(curV)), curV; kwargs...)
    nexpl = default_nexpl_steps(xpl)
    np    = NRSTProblem(                                         # instantiate an NRSTProblem
        tm, N, betas, similar(betas), use_mean, reject_big_vs, 
        log_grid, fill(nexpl, N), fill(params(xpl), N)
    )
    ip    = MVector(zero(N), one(N))
    NRSTSampler(np, xpl, x, ip, curV)
end

init_grid(N::Int) = collect(range(0.,1.,N+1)) # grid initialization
initx(pre_x, args...) = pre_x                 # default x initialization

# safe initialization for arrays with float entries
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TF}, rng::AbstractRNG) where {TF<:AbstractFloat}
    x = rand(rng, Uniform(-one(TF), one(TF)), size(pre_x))
    x .* (sign.(x) .* sign.(pre_x)) # quick and dirty way to respect sign constraints 
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

# reset state by sampling from the renewal measure
# we know that iprop=-1 is rejected always, so we can anticipate that.
function renew!(ns::NRSTSampler{T,I}, rng::AbstractRNG) where {T,I}
    ns.ip[1] = zero(I)
    ns.ip[2] = one(I)
    refreshx!(ns, rng)
end

# handling last tour step
function save_last_step_tour!(ns::NRSTSampler{T,I,K}, tr; kwargs...) where {T,I,K}
    save_pre_step!(ns, tr; kwargs...)               # store state at atom
    save_post_step!(ns, tr, one(K), K(NaN), one(I)) # we know that (-1,-1) would be rejected if attempted so we store this. also, the expl step would not use an explorer; thus the NaN. Finally, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end

