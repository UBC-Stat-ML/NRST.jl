##############################################################################
# methods for simulated tempering samplers
##############################################################################

abstract type AbstractSTSampler{T,TI<:Int,TF<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem} <: RegenerativeSampler{T,TI,TF} end

# grid initialization
init_grid(N::Int) = collect(range(0.,1.,N+1)) # vcat(0., range(1e-8, 1., N))

# safe initialization for arrays with float entries
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TF}, rng::AbstractRNG) where {TF<:AbstractFloat}
    x = rand(rng, Uniform(-one(TF), one(TF)), size(pre_x))
    x .* (sign.(x) .* sign.(pre_x)) # quick and dirty way to respect sign constraints 
end

# copy-constructor, using a given AbstractSTSampler (usually already tuned)
Base.copy(st::TS) where {TS <: AbstractSTSampler} = TS(copyfields(st)...)
function copyfields(st::AbstractSTSampler)
    newtm = copy(st.np.tm)                   # the only element in np that we (may) need to copy
    newnp = NRSTProblem(st.np, newtm)        # build new Problem from the old one but using new tm
    newx  = copy(st.x)
    ncurV = Ref(st.curV[])
    nuxpl = copy(st.xpl, newtm, newx, ncurV) # copy st.xpl sharing stuff with the new sampler
    return newnp, nuxpl, newx, MVector(0,1), ncurV
end

# for all ST samplers we can use an NRSTTrace
function get_trace(st::TS, args...) where {T,TI,TF,TS <: AbstractSTSampler{T,TI,TF}}
    NRSTTrace(T, st.np.N, TF, args...)
end

###############################################################################
# sampling methods
###############################################################################

# sample from ref (possibly using rejection to avoid V=inf) and update curV accordingly
function refreshx!(st::AbstractSTSampler, rng::AbstractRNG)
    st.curV[] = randrefmayreject!(st.np.tm, rng, st.x, st.np.reject_big_vs)
end

# exploration step
function expl_step!(st::TS, rng::AbstractRNG) where {T,I,K,TS<:AbstractSTSampler{T,I,K}}
    @unpack np,xpl,ip = st
    xplap = one(K)
    if ip[1] == zero(I)
        refreshx!(st, rng)                            # sample from ref
    else
        β      = np.betas[ip[1]+1]                    # get the β for the level
        nexpl  = np.nexpls[ip[1]]                     # get number of exploration steps needed at this level
        params = np.xplpars[ip[1]]                    # get explorer params for this level 
        xplap  = explore!(xpl, rng, β, params, nexpl) # explore for nexpl steps. note: st.x and st.curV are shared with xpl
    end
    return xplap
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

# methods for storing results
function save_pre_step!(st::AbstractSTSampler, tr::NRSTTrace, n::Int; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && copyto!(trX[n],st.x) # needs copy o.w. we get a ref to st.x
    copyto!(trIP[n], st.ip)         # same
    trV[n] = st.curV[]
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

#######################################
# RegenerativeSampler interface
#######################################

function save_pre_step!(st::AbstractSTSampler, tr::NRSTTrace; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && push!(trX, copy(st.x))          # needs copy o.w. pushes a ref to st.x
    push!(trIP, copy(st.ip))                   # same
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
