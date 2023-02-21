###############################################################################
# Slice samplers à la Neal (2003)
###############################################################################

abstract type SliceSampler{TTM<:TemperedModel,TF<:AbstractFloat,TV<:AbstractVector{TF},TI<:Int} <: ExplorationKernel end

#######################################
# construction and copying
#######################################

params(ss::SliceSampler) = (w=ss.w[],)                     # get params as namedtuple
function set_params!(ss::SliceSampler, params::NamedTuple) # set sigma from a NamedTuple
    ss.w[] = params.w
end

# low level copy constructor
# note: this is called only from the generic method for copying ExplorationKernels
function Base.copy(ss::SliceSampler, args...)
    SliceSampler(args..., Ref(ss.w[]), ss.p)
end

default_nexpl_steps(ss::SliceSampler) = one(ss.p)

#######################################
# sampling methods
#######################################

# compute all potentials by temporarilly changing x[i]
function potentials(ss::SliceSampler, i::Int, newxi::Real)
    xi      = ss.x[i]
    ss.x[i] = newxi
    ps      = potentials(ss, ss.x)
    ss.x[i] = xi
    return ps
end

# check if point is in slice by looking at its potential
# note: y < πβ(x) <=> newv > -log(pi_beta(x)) =: Vβ(x)
Vβ(ps::Tuple) = last(ps)                          # extract the tempered potential from a tuple of potentials
in_slice(newv::Real, ps::Tuple) = (newv > Vβ(ps))

# build a slice
function build_slice(ss::SliceSampler, rng::AbstractRNG, i::Int, newv::Real)
    grow_slice(ss, rng, i, init_slice(ss, rng, i)..., newv)
end

# initialize slice and compute potentials at both endpoints
function init_slice(ss::SliceSampler, rng::AbstractRNG, i::Int)
    xi  = ss.x[i]              # current value at position i
    L   = xi - ss.w[]*rand(rng)
    R   = L + ss.w[]
    Lps = potentials(ss, i, L) # note: (L|R)ps are tuples
    Rps = potentials(ss, i, R)
    # @debug "init_slice: (L,R)=($L,$R)"
    L, R, Lps, Rps
end

# select point by shrinking (Alg. in Fig 5)
function shrink_slice(ss::SliceSampler, rng::AbstractRNG, i, L, R, Lps, Rps, newv)
    xi    = ss.x[i]
    newxi = xi                           # init with the current point
    newps = potentials(ss)               # init with the potentials at the current point
    bL    = L
    bR    = R
    nvs   = 0                            # counts number of V(x) evaluations
    tol   = 10eps(typeof(xi))
    while true
        if bR-bL < tol                   # failsafe for degenerate distributions and potential rounding issues. See e.g. here: https://github.com/UBC-Stat-ML/blangSDK/blob/e9f57ad63476a18added1dd97e761d5f5b26adf0/src/main/java/blang/mcmc/RealSliceSampler.java#L109 
            newxi = xi
            newps = potentials(ss)
            break
        end
        newxi = bL + (bR-bL)*rand(rng)   # select a point in (bL,bR) at random
        newps = potentials(ss, i, newxi) # compute the potentials at that point
        nvs  += 1
        if in_slice(newv, newps)
            acc,nv = is_acceptable(ss, i, newxi, L, R, Lps, Rps, newv)
            # @debug "shrink_slice: (bL,bR)=($bL,$bR), newxi=$newxi => acc=$(acc)!"
            nvs   += nv
            acc && break
        end
        newxi < xi ? (bL = newxi) : (bR = newxi)            
    end
    return newxi, newps, nvs
end

# by default all proposals are accepted
is_acceptable(::SliceSampler{TTM,TF,TV,TI}, args...) where {TTM,TF,TV,TI} = (true, zero(TI))

# univariate step
function step!(ss::SliceSampler, rng::AbstractRNG, i::Int)
    newv    = ss.curVβ[] + randexp(rng)  # draw a new slice. note: y = pibeta(x)*U(0,1) <=> -log(y) =: newv = Vβ(x) + Exp(1)
    L,R,Lps,Rps,nv = build_slice(ss, rng, i, newv)
    nvs     = nv + 2                     # number of V(x) evaluations = 2 (init_slice) + nv (grow_slice)
    newxi,newps,nv = shrink_slice(ss, rng, i, L, R, Lps, Rps, newv)
    nvs    += nv
    ss.x[i] = newxi                      # update state
    set_potentials!(ss, newps...)        # update potentials
    return nvs
end

# multivariate step: Gibbs update
function step!(ss::SliceSampler, rng::AbstractRNG)
    nvs = zero(ss.p)                     # number of V(x) evaluations
    for i in eachindex(ss.x)
        nvs += step!(ss, rng, i)
    end
    return one(eltype(ss.curV)), nvs     # return "acceptance probability" and number of V evals
end

###############################################################################
# Slice sampler with stepping out strategy (schema 3 in Neal (2003))
###############################################################################

struct SliceSamplerStepping{TTM,TF,TV,TI} <: SliceSampler{TTM,TF,TV,TI}
    # fields common to every ExplorationKernel
    tm::TTM                                    # TemperedModel
    x::TV                                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{TF}                    # current beta
    curVref::Base.RefValue{TF}                 # current reference potential
    curV::Base.RefValue{TF}                    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{TF}                   # current tempered potential
    # idiosyncratic fields
    w::Base.RefValue{TF}                       # initial slice width
    p::TI                                      # max slice width = pw (this is m in Neal's notation)
end

# outer constructor
function SliceSamplerStepping(tm, x, curβ, curVref, curV, curVβ; w=4.0, p=20, kwargs...)
    SliceSamplerStepping(tm, x, curβ, curVref, curV, curVβ, Ref(w), p)
end

#######################################
# sampling methods
#######################################

# grow slice using the stepping out approach (Alg. in Fig 3)
function grow_slice(ss::SliceSamplerStepping, rng::AbstractRNG, i, L, R, Lps, Rps, newv)
    w = ss.w[]
    p = ss.p
    J = floor(Int, p*rand(rng))          # max number of steps to the left
    K = (p - one(p)) - J                 # max number of steps to the right
    while J > 0 && in_slice(newv, Lps)
        L  -= w
        Lps = potentials(ss, i, L)
        J  -= 1 
    end
    while K > 0 && in_slice(newv, Rps)
        R  += w
        Rps = potentials(ss, i, R)
        K  -= 1 
    end
    L, R, Lps, Rps, (p - one(p)) - (J+K) # number of V(x) evaluations
end

###############################################################################
# Slice sampler with doubling strategy (schema 4 in Neal (2003))
###############################################################################

struct SliceSamplerDoubling{TTM,TF,TV,TI} <: SliceSampler{TTM,TF,TV,TI}
    # fields common to every ExplorationKernel
    tm::TTM                                    # TemperedModel
    x::TV                                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{TF}                    # current beta
    curVref::Base.RefValue{TF}                 # current reference potential
    curV::Base.RefValue{TF}                    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{TF}                   # current tempered potential
    # idiosyncratic fields
    w::Base.RefValue{TF}                       # initial slice width
    p::TI                                      # max slice width = w2^p
end

# outer constructor
function SliceSamplerDoubling(tm, x, curβ, curVref, curV, curVβ; w=14.0, p=20, kwargs...)
    SliceSamplerDoubling(tm, x, curβ, curVref, curV, curVβ, Ref(w), p)
end

#######################################
# sampling methods
#######################################

# grow slice using the doubling approach (Alg. in Fig 4)
function grow_slice(ss::SliceSamplerDoubling, rng::AbstractRNG, i, L, R, Lps, Rps, newv)
    k = zero(ss.p)
    while (in_slice(newv, Lps) || in_slice(newv, Rps)) && k < ss.p
        grow_left = rand(rng, Bool)
        if grow_left
            L  -= (R-L)
            Lps = potentials(ss, i, L) 
        else
            R  += (R-L)
            Rps = potentials(ss, i, R) 
        end
        k += 1
        # @debug "grow_slice: (k=$k): (L,R)=($L,$R)"
    end
    L, R, Lps, Rps, k # k == number of V(x) evaluations
end

# check acceptability of candidate point (Alg. in Fig 6)
function is_acceptable(ss::SliceSamplerDoubling, i, newxi, L, R, Lps, Rps, newv)
    xi   = ss.x[i]
    w    = ss.w[]
    hL   = L
    hR   = R
    hLps = Lps
    hRps = Rps
    acc  = true
    nvs  = 0
    while hR-hL > 1.1*w
        M = (hL+hR)/2.0
        D = (xi < M && newxi >= M) || (xi >= M && newxi < M) # are xi and newxi on Different sides wrt M?
        if newxi < M
            hR   = M
            hRps = potentials(ss, i, hR)
        else
            hL   = M
            hLps = potentials(ss, i, hL)
        end
        nvs += 1
        if D && !in_slice(newv, hLps) && !in_slice(newv, hRps)
            acc = false
            break
        end
    end
    return acc, nvs
end
