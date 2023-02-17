###############################################################################
# Slice sampler with doubling strategy (schema 4 in Neal (2003))
###############################################################################

struct SliceSampler{TTM<:TemperedModel,TF<:AbstractFloat,TV<:AbstractVector{TF},TI<:Int} <: ExplorationKernel
    # fields common to every ExplorationKernel
    tm::TTM                    # TemperedModel
    x::TV                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{TF}    # current beta
    curVref::Base.RefValue{TF} # current reference potential
    curV::Base.RefValue{TF}    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{TF}   # current tempered potential
    # idiosyncratic fields
    xprop::TV                  # temp storage for calculating potentials during sampling
    w::Base.RefValue{TF}       # initial slice width
    p::TI                      # max slice width = w2^p
end

#######################################
# construction and copying
#######################################

# outer constructor
function SliceSampler(tm, x, curβ, curVref, curV, curVβ; w=10.0, p=20)
    SliceSampler(tm, x, curβ, curVref, curV, curVβ, similar(x), Ref(w), p)
end
params(ss::SliceSampler) = (w=ss.w[],)                  # get params as namedtuple
function set_params!(ss::SliceSampler, params::NamedTuple) # set sigma from a NamedTuple
    ss.w[] = params.w
end

# copy constructor
# note: this is called only from the generic method for copying ExplorationKernels
function Base.copy(ss::SliceSampler, args...)
    SliceSampler(args..., similar(x), Ref(ss.w[]), ss.p)
end

#######################################
# sampling methods
#######################################

# compute all potentials by setting xprop[i] = newxi
function potentials(ss::SliceSampler, i::Int, newxi::Real)
    ss.xprop[i] = newxi
    potentials(ss, ss.xprop)
end
Vβ(ps::Tuple) = ps[3] # extract the tempered potential from a tuple of potentials

# build a slice
function build_slice(ss::SliceSampler, rng::AbstractRNG, i::Int)
    grow_slice(ss, rng, i, init_slice(ss, rng, i)...)
end

# initialize slice and compute potentials at both endpoints
function init_slice(ss::SliceSampler, rng::AbstractRNG, i::Int)
    xi  = ss.x[i]              # current value at position i
    L   = xi - ss.w[]*rand(rng)
    R   = L + ss.w[]
    Lps = potentials(ss, i, L) # note: (L|R)ps are tuples
    Rps = potentials(ss, i, R)
    L, R, Lps, Rps
end

# grow slice using the doubling approach (Alg. in Fig 4)
# y < πβ(x) <=> vβ₀ := -log(y) > -log(pi_beta(x)) =: Vβ(x) 
in_slice(ss::SliceSampler, ps::Tuple) = (ss.curVβ[] > Vβ(ps))
function grow_slice(ss::SliceSampler, rng::AbstractRNG, i, L, R, Lps, Rps)
    k = zero(ss.p)
    while (in_slice(ss, Lps) || in_slice(ss, Rps)) && k < ss.p
        grow_left = rand(rng) < 0.5
        if grow_left
            L  -= (R-L)
            Lps = potentials(ss, i, L) 
        else
            R  += (R-L)
            Rps = potentials(ss, i, R) 
        end
        k += 1
    end
    L, R, Lps, Rps, k # k == number of V(x) evaluations
end

# select point by shrinking (Alg. in Fig 5)
function shrink_slice(ss::SliceSampler, rng::AbstractRNG, i, L, R, Lps, Rps)
    xi    = ss.x[i]
    newxi = xi                           # init with the current point
    newps = potentials(ss)               # init with the potentials at the current point
    bL    = L
    bR    = R
    nvs   = 0                            # counts number of V(x) evaluations
    while true
        newxi = bL + (bR-bL)*rand(rng)   # select a point in (bL,bR) at random
        newps = potentials(ss, i, newxi) # compute the potentials at that point
        nvs  += 1
        if in_slice(ss, newps)
            acc,nv = is_acceptable(ss, i, newxi, L, R, Lps, Rps)
            nvs   += nv
            acc && break
        end
        newxi < xi ? (bL = newxi) : (bR = newxi)            
    end
    return newxi, newps, nvs
end

# check acceptability of candidate point (Alg. in Fig 6)
function is_acceptable(ss::SliceSampler, i, newxi, L, R, Lps, Rps)
    xi   = ss.x[i]
    w    = ss.w[]
    hL   = L
    hR   = R
    hLps = Lps
    hRps = Rps
    acc  = true
    nvs  = 0
    while hR-hL > 1.1*w
        M = 0.5(hL+hR)
        D = (xi < M && newxi >= M) || (xi >= M && newxi < M) # are xi and newxi on Different sides wrt M?
        if newxi < M
            hR   = M
            hRps = potentials(ss, i, hR)
        else
            hL   = M
            hLps = potentials(ss, i, hL)
        end
        nvs += 1
        if D && !in_slice(ss, hLps) && !in_slice(ss, hRps)
            acc = false
            break
        end
    end
    return acc, nvs
end

# univariate step
function step!(ss::SliceSampler, rng::AbstractRNG, i::Int)
    L,R,Lps,Rps,nv = build_slice(ss, rng, i)
    nvs  = nv + 2                 # number of V(x) evaluations
    newxi,newps,nv = shrink_slice(ss, rng, i, L, R, Lps, Rps)
    nvs += nv
    ss.xprop[i]    = newxi        # update proposal (actual state is updated after looping over all i) 
    set_potentials!(ss, newps...) # update potentials
    return nvs
end

# multivariate step
function step!(ss::SliceSampler, rng::AbstractRNG)
    nvs = 0                       # number of V(x) evaluations
    copyto!(ss.xprop, ss.x)       # init xprop with current x
    for i in eachindex(ss.x)
        nvs += step!(ss, rng, i)
    end
    copyto!(ss.x, ss.xprop)       # update state (potentials are updated inside the loop)
    return nvs
end