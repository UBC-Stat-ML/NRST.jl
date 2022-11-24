###############################################################################
# NRSTSampler
###############################################################################

struct NRSTSampler{T,I<:Int,K<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem} <: RegenerativeSampler{T,I,K,TXp,TProb}
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

# grid initialization
init_grid(N::Int) = collect(range(0.,1.,N+1)) # vcat(0., range(1e-8, 1., N))

# safe initialization for arrays with float entries
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TF}, rng::AbstractRNG) where {TF<:AbstractFloat}
    x = rand(rng, Uniform(-one(TF), one(TF)), size(pre_x))
    x .* (sign.(x) .* sign.(pre_x)) # quick and dirty way to respect sign constraints 
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    tm = SimpleTemperedModel(V, Vref, randref)
    NRSTSampler(tm,args...;kwargs...)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
function Base.copy(ns::NRSTSampler)
    newtm = copy(ns.np.tm)                   # the only element in np that we (may) need to copy
    newnp = NRSTProblem(ns.np, newtm)        # build new Problem from the old one but using new tm
    newx  = copy(ns.x)
    ncurV = Ref(ns.curV[])
    nuxpl = copy(ns.xpl, newtm, newx, ncurV) # copy ns.xpl sharing stuff with the new sampler
    NRSTSampler(newnp, nuxpl, newx, MVector(0,1), ncurV)
end

###############################################################################
# sampling methods
###############################################################################

# communication step: the acceptance ratio is given by
# A = [pi^{(i+eps)}(x)/pi^{(i)}(x)] [p_{i+eps}/p_i]
# = [Z(i)/Z(i+eps)][exp{-b_{i+eps}V(x)}exp{b_{i}V(x)}][Z(i+eps)/Z(i)][exp{c_{i+eps}exp{-c_{i}}]
# = exp{-[b_{i+eps} - b_{i}]V(x) + c_{i+eps} -c_{i}}
# = exp{ -( [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i}) ) }
# = exp(-nlaccr)
# where nlaccr := -log(A) = [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i})
# To get the rejection probability from nlaccr:
# ap = min(1.,A) = min(1.,exp(-nlaccr)) = exp(min(0.,-nlaccr)) = exp(-max(0.,nlaccr))
# => rp = 1-ap = 1-exp(-max(0.,nlaccr)) = -[exp(-max(0.,nlaccr))-1] = -expm1(-max(0.,nlaccr))
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
        i      = ip[1]                          # current index
        nlaccr = (betas[iprop+1]-betas[i+1])*curV[] - (c[iprop+1]-c[i+1])
        acc    = nlaccr < randexp(rng)          # accept? Note: U<A <=> A>U <=> -log(A) < -log(U) ~ Exp(1) 
        if acc
            ip[1] = iprop                       # move
        else
            ip[2] = -ip[2]                      # flip direction
        end
    end
    rp = -expm1(-max(zero(K), nlaccr))
    return rp
end

# exploration step
function expl_step!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack np,xpl,ip = ns
    xplap = one(K)
    if ip[1] == zero(I)
        refreshx!(ns, rng)                            # sample from ref (possibly using rejection to avoid V=inf) and update curV accordingly
    else
        β      = np.betas[ip[1]+1]                    # get the β for the level
        nexpl  = np.nexpls[ip[1]]                     # get number of exploration steps needed at this level
        params = np.xplpars[ip[1]]                    # get explorer params for this level 
        xplap  = explore!(xpl, rng, β, params, nexpl) # explore for nexpl steps. note: ns.x and ns.curV are shared with xpl
    end
    return xplap
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler, rng::AbstractRNG)
    rp    = comm_step!(ns, rng) # returns rejection probability
    xplap = expl_step!(ns, rng) # returns explorers' acceptance probability
    return rp, xplap
end

# run for fixed number of steps
function run!(ns::NRSTSampler, rng::AbstractRNG, tr::NRSTTrace)
    @unpack trX, trIP, trV, trRP, trXplAP = tr
    nsteps = length(trV)
    for n in 1:nsteps
        trX[n]    = copy(ns.x)                 # needs copy o.w. pushes a ref to ns.x
        trIP[n]   = copy(ns.ip)
        trV[n]    = ns.curV[]
        rp, xplap = step!(ns, rng)
        trRP[n]   = rp                         # note: since trIP[n] was stored before step!, trRP[n] is rej prob of swap **initiated** from trIP[n]
        l         = ns.ip[1]
        l >= 1 && push!(trXplAP[l], xplap)     # note: since comm preceeds expl, we're correctly storing the acc prob of the most recent state
    end
end
function run!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG; nsteps::Int) where {T,I,K}
    tr = NRSTTrace(T, ns.np.N, K, nsteps)
    run!(ns, rng, tr)
    return tr
end

#######################################
# RegenerativeSampler interface
#######################################

isinatom(ns::NRSTSampler{T,I}) where {T,I} = (ns.ip[1]==zero(I) && ns.ip[2]==-one(I))

function refreshx!(ns::NRSTSampler, rng::AbstractRNG)
    ns.curV[] = randrefmayreject!(ns.np.tm, rng, ns.x, ns.np.reject_big_vs)
end

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler{T,I}, rng::AbstractRNG) where {T,I}
    refreshx!(ns, rng)
    ns.ip[1] = zero(I)
    ns.ip[2] = one(I)
end

function save_pre_step!(ns::NRSTSampler, tr::NRSTTrace; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && push!(trX, copy(ns.x))          # needs copy o.w. pushes a ref to ns.x
    push!(trIP, copy(ns.ip))                   # same
    push!(trV, ns.curV[])
    return
end
function save_post_step!(
    ns::NRSTSampler,
    tr::NRSTTrace,
    rp::AbstractFloat, 
    xplap::AbstractFloat
    )
    push!(tr.trRP, rp)
    l = ns.ip[1]
    l >= 1 && push!(tr.trXplAP[l], xplap)
    return
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
