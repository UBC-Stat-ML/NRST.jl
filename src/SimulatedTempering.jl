##############################################################################
# methods for simulated tempering samplers
##############################################################################

abstract type AbstractSTSampler{T,TI<:Int,TF<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem} <: RegenerativeSampler{T,TI,TF} end

# copy-constructor, using a given AbstractSTSampler (usually already tuned)
Base.copy(st::TS) where {TS <: AbstractSTSampler} = TS(copyfields(st)...)
function copyfields(st::TST) where {T,TI,TF,TXp,TST<:AbstractSTSampler{T,TI,TF,TXp}}
    newtm = copy(st.np.tm)                         # the only element in np that we (may) need to copy
    newnp = NRSTProblem(st.np, newtm)              # build new Problem from the old one but using new tm
    newx  = copy(st.x)
    ncurV = Ref(st.curV[])
    nuxpl = TXp.name.wrapper(                      # build a new explorer sharing stuff with the new sampler. uses # hack: https://discourse.julialang.org/t/get-generic-constructor-of-parametric-type/57189/2
        newtm, newx, st.xpl.curβ[], ncurV
    )
    return newnp, nuxpl, newx, MVector(0,1), ncurV
end

#######################################
# traces
#######################################

# generic constructor for each AbstractTrace when given an AbstractSTSampler
function (::Type{TTrace})(st::AbstractSTSampler) where {TTrace <: AbstractTrace}
    TTrace(typeof(st.x), st.np.N, eltype(st.curV))
end

# default trace for ST samplers
get_trace(st::AbstractSTSampler) = ConstCostTrace(st)

###############################################################################
# sampling methods
###############################################################################

# NRST step = comm_step ∘ expl_step => (X,0,-1) is atom
# default method. works for NRST, SH16, and others
function step!(st::AbstractSTSampler, rng::AbstractRNG)
    rp      = comm_step!(st, rng) # returns rejection probability
    xap,nvs = expl_step!(st, rng) # returns explorers' acceptance probability and number of V(x) evaluations
    return rp, xap, nvs
end

# sample from ref (possibly using rejection to avoid V=inf) and update curV accordingly
function refreshx!(st::AbstractSTSampler, rng::AbstractRNG)
    v, nvs = randrefmayreject!(st.np.tm, rng, st.x, st.np.reject_big_vs)
    st.curV[] = v
    nvs
end

# exploration step
function expl_step!(st::TS, rng::AbstractRNG) where {T,I,K,TS<:AbstractSTSampler{T,I,K}}
    @unpack np,xpl,ip = st
    xap = one(K)
    if ip[1] == zero(I)
        nvs    = refreshx!(st, rng)                   # sample from ref, return number of V evals (>1 if rejections occured) 
    else
        β      = np.betas[ip[1]+1]                    # get the β for the level
        nexpl  = np.nexpls[ip[1]]                     # get number of exploration steps needed at this level
        params = np.xplpars[ip[1]]                    # get explorer params for this level 
        xap,nvs= explore!(xpl, rng, β, params, nexpl) # explore for nexpl steps. note: st.x and st.curV are shared with xpl
    end
    return xap,nvs
end

# negative log acceptance ratio (nlar) for communication step
# note: does not account for Hastings ratio, which changes with the algo
get_nlar(β₀,β₁,c₀,c₁,v) = (β₁-β₀)*v - (c₁-c₀)

# convert nlar to rejection probability
# note: the acceptance ratio is given by
# A = [pi^{(i+eps)}(x)/pi^{(i)}(x)] [p_{i+eps}/p_i]
# = [Z(i)/Z(i+eps)][exp{-b_{i+eps}V(x)}exp{b_{i}V(x)}][Z(i+eps)/Z(i)][exp{c_{i+eps}exp{-c_{i}}]
# = exp{-[b_{i+eps} - b_{i}]V(x) + c_{i+eps} -c_{i}}
# = exp{ -( [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i}) ) }
# = exp(-nlaccr)
# where nlaccr := -log(A) = [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i})
# To get the rejection probability from nlaccr:
# ap = min(1.,A) = min(1.,exp(-nlaccr)) = exp(min(0.,-nlaccr)) = exp(-max(0.,nlaccr))
# => rp = 1-ap = 1-exp(-max(0.,nlaccr)) = -[exp(-max(0.,nlaccr))-1] = -expm1(-max(0.,nlaccr))
nlar_2_rp(nlar) = -expm1(-max(zero(nlar), nlar))
nlar_2_ap(nlar) = exp(-max(zero(nlar), nlar))

#######################################
# methods for storing results in traces
#######################################

function save_pre_step!(st::AbstractSTSampler, tr::NRSTTrace, n::Int; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && (trX[n] = copy(st.x)) # needs copy o.w. we get a ref to st.x
    trIP[n] = copy(st.ip)            # same
    trV[n]  = st.curV[]
    return
end
function save_post_step!(
    st::AbstractSTSampler,
    tr::NRSTTrace,
    n::Int,
    rp::AbstractFloat, 
    xplap::AbstractFloat
    )
    @unpack trRP, trXplAP = tr
    trRP[n]   = rp                         # note: since trIP[n] was stored before step!, trRP[n] is rej prob of swap **initiated** from trIP[n]
    l         = st.ip[1]
    l >= 1 && push!(trXplAP[l], xplap)     # note: since comm preceeds expl, we're correctly storing the acc prob of the most recent state
    return
end

##############################################################################
# RegenerativeSampler interface
##############################################################################

# check if state is in the atom
# default method. works for NRST, SH16, and others
function isinatom(st::AbstractSTSampler{T,I}) where {T,I}
    first(st.ip)==zero(I) && last(st.ip)==-one(I)
end

#######################################
# methods for storing results in traces
#######################################

# NRSTTrace
function save_pre_step!(st::AbstractSTSampler, tr::NRSTTrace; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && push!(trX, copy(st.x)) # needs copy o.w. pushes a ref to st.x
    push!(trIP, st.ip)                # no "copy" because implicit conversion from MVector to SVector does the copying
    push!(trV, st.curV[])
    return
end
function save_post_step!(
    st::AbstractSTSampler,
    tr::NRSTTrace,
    rp::AbstractFloat, 
    xplap::AbstractFloat
    )
    push!(tr.trRP, rp)
    l = st.ip[1]
    l >= 1 && push!(tr.trXplAP[l], xplap)
    return
end

# IPRPTrace
function save_pre_step!(st::AbstractSTSampler, tr::IPRPTrace)
    push!(tr.trIP, st.ip) # no "copy" because implicit conversion from MVector to SVector does the copying
    return
end
function save_post_step!(::AbstractSTSampler, tr::IPRPTrace, rp, args...)
    push!(tr.trRP, rp)
    return
end

# ConstCostTrace
function save_pre_step!(::AbstractSTSampler, ::ConstCostTrace) end
function save_post_step!(st::AbstractSTSampler, tr::ConstCostTrace, _, _, nvs)
    tr.n_steps[] += 1
    i = first(st.ip)
    i == st.np.N && (tr.n_vis_top[] += 1)
    tr.n_v_evals[] += nvs
    return
end

#######################################
# parallel runs
#######################################

# multithreading method with automatic determination of required number of tours
const DEFAULT_α      = .95  # probability of the mean of indicators to be inside interval
const DEFAULT_δ      = .10  # half-width of interval
const DEFAULT_TE_min = 1e-6 # truncate TE's below this value
function min_ntours_TE(TE, α=DEFAULT_α, δ=DEFAULT_δ, TE_min=DEFAULT_TE_min)
    ceil(Int, (4/max(TE_min,TE)) * abs2(norminvcdf((1+α)/2) / δ))
end
const DEFAULT_MAX_TOURS = min_ntours_TE(0.)

# computes ntours from TE
function parallel_run(
    st::AbstractSTSampler,
    rng::SplittableRandom,
    trace_template::AbstractTrace;
    TE::AbstractFloat = NaN,
    ntours::Int       = 0,
    α::AbstractFloat  = DEFAULT_α,
    δ::AbstractFloat  = DEFAULT_δ,
    kwargs...
    )
    isfinite(TE) && iszero(ntours) && (ntours=min_ntours_TE(TE,α,δ))
    Base.@invoke parallel_run(
        st::RegenerativeSampler, rng::SplittableRandom, trace_template::AbstractTrace;
        ntours = ntours, kwargs...
    )
end
