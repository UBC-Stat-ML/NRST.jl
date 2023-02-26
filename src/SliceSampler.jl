###############################################################################
# Slice samplers à la Neal (2003)
###############################################################################

abstract type SliceSamplerStrategy end
struct Doubling <: SliceSamplerStrategy end
struct SteppingOut <: SliceSamplerStrategy end

struct SliceSampler{SSS<:SliceSamplerStrategy,TTM<:TemperedModel,TF<:AbstractFloat,TV<:AbstractVector{TF},TI<:Int} <: ExplorationKernel
    # fields common to every ExplorationKernel
    tm::TTM                                    # TemperedModel
    x::TV                                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{TF}                    # current beta
    curVref::Base.RefValue{TF}                 # current reference potential
    curV::Base.RefValue{TF}                    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{TF}                   # current tempered potential
    # idiosyncratic fields
    w::Base.RefValue{TF}                       # initial slice width
    p::TI                                      # max slice width is wp if using SteppingOut or w2^p if using doubling
end

#######################################
# construction and initialization
#######################################

# outer constructor: mix common explorer fields with idiosyncratic
function SliceSampler{SSS,TTM,TF,TV,TI}(
    tm::TTM,
    x::TV,
    curβ::TRef,
    curVref::TRef,
    curV::TRef,
    curVβ::TRef;
    w = default_w(SSS),
    p,
    kwargs...
    ) where {SSS,TTM,TF,TV,TI,TRef<:Base.RefValue{TF}}
    SliceSampler{SSS,TTM,TF,TV,TI}(tm, x, curβ, curVref, curV, curVβ, Ref(TF(w)), TI(p))
end

# outer constructor: take idiosyncratic parameters from a reference SliceSampler
function SliceSampler{SSS,TTM,TF,TV,TI}(oldss::SliceSampler, args...) where {SSS,TTM,TF,TV,TI}
    SliceSampler{SSS,TTM,TF,TV,TI}(args..., Ref(oldss.w[]), TI(oldss.p))
end

# outer constructor: infer missing parametric values
function SliceSampler{SSS}(
    tm::TemperedModel,
    x,
    curβ::Base.RefValue{<:AbstractFloat}, 
    args...;
    p::Int = 20, 
    kwargs...
    ) where {SSS<:SliceSamplerStrategy}
    SliceSampler{SSS,typeof(tm),eltype(curβ),typeof(x),typeof(p)}(
        tm, x, curβ, args...; p=p, kwargs...
    )
end

# sugar
const SliceSamplerDoubling    = SliceSampler{Doubling}
const SliceSamplerSteppingOut = SliceSampler{SteppingOut}

# default window size
default_w(::Type{Doubling})    = 14.0 # ~min-cost-optimal for N(0,1)
default_w(::Type{SteppingOut}) = 4.0  # ~min-cost-optimal for N(0,1)

#######################################
# utils
#######################################

params(ss::SliceSampler) = (w=ss.w[],)                     # get params as namedtuple
function set_params!(ss::SliceSampler, params::NamedTuple) # set sigma from a NamedTuple
    ss.w[] = params.w
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
    # @debug "init_slice: (L, R) = ($L, $R)"
    L, R, Lps, Rps
end

# grow slice using the stepping out approach (Alg. in Fig 3)
function grow_slice(ss::SliceSampler{SteppingOut}, rng::AbstractRNG, i, L, R, Lps, Rps, newv)
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

# grow slice using the doubling approach (Alg. in Fig 4)
function grow_slice(ss::SliceSampler{Doubling}, rng::AbstractRNG, i, L, R, Lps, Rps, newv)
    k = zero(ss.p)
    while (in_slice(newv, Lps) || in_slice(newv, Rps)) && k < ss.p
        # print("grow_slice: i=$i, k=$k, L=$L, R=$R, grow_left="); flush(stdout)
        grow_left = rand(rng, Bool)
        # println(grow_left); flush(stdout)
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
function shrink_slice(
    ss::SliceSampler{SSS,TTM,TF}, rng::AbstractRNG, i, L, R, Lps, Rps, newv
    ) where {SSS,TTM,TF}
    xi    = ss.x[i]
    newxi = xi                                               # init with the current point
    newps = potentials(ss)                                   # init with the potentials at the current point
    bL    = L
    bR    = R
    nvs   = 0                                                # counts number of V(x) evaluations
    atol  = 100eps(TF)
    rtol  = sqrt(eps(TF))
    while true
        # print("\t\tshrink_slice: (bL, bR) = ($bL, $bR), "); flush(stdout)
        if bR-bL < atol || bL/bR > 1-rtol                    # failsafe for infinite loops due to degenerate distributions and potential rounding issues. atol problem seen for Doubling with HierarchicalModel seed 6872 γ=1.5 median. rtol problem seen for Doubling with HierarchicalModel seed 2986 γ=2.0 median. See also e.g. here: https://github.com/UBC-Stat-ML/blangSDK/blob/e9f57ad63476a18added1dd97e761d5f5b26adf0/src/main/java/blang/mcmc/RealSliceSampler.java#L109 
            newxi = xi
            newps = potentials(ss)
            # println("=> too close together => output newxi=xi"); flush(stdout)
            break
        end
        newxi = bL + (bR-bL)*rand(rng)                       # select a point in (bL,bR) at random
        # println("newxi = $newxi."); flush(stdout)
        newps = potentials(ss, i, newxi)                     # compute the potentials at that point
        nvs  += 1
        if in_slice(newv, newps)
            # println("\t\tshrink_slice: newxi in slice! checking acceptability..."); flush(stdout)
            acc,nv = is_acceptable(ss, i, newxi, L, R, Lps, Rps, newv)
            nvs   += nv
            # println("\t\tshrink_slice: accepted!"); flush(stdout)
            acc && break
        end
        # println("\t\tshrink_slice: rejected!"); flush(stdout)
        newxi < xi ? (bL = newxi) : (bR = newxi)            
    end
    return newxi, newps, nvs
end

# all proposals are accepted when using stepping out
is_acceptable(::SliceSampler{SteppingOut}, args...) = (true, 0)

# check acceptability of candidate point (Alg. in Fig 6)
function is_acceptable(ss::SliceSampler{Doubling}, i, newxi, L, R, Lps, Rps, newv)
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
        # println("\t\t\tis_acceptable: k=$nvs: D=$D, (hL, newxi, hR) = ($hL, $newxi, $hR), (VβhL, newv, VβhR) = ($(Vβ(hLps)), $newv, $(Vβ(hRps)))")
        if D && !in_slice(newv, hLps) && !in_slice(newv, hRps)
            acc = false
            break
        end
    end
    return acc, nvs
end

# univariate step
function step!(ss::SliceSampler, rng::AbstractRNG, i::Int)
    newv    = ss.curVβ[] + randexp(rng)  # draw a new slice. note: y = pibeta(x)*U(0,1) <=> -log(y) =: newv = Vβ(x) + Exp(1)
    # print("\tbuilding slice..."); flush(stdout)
    L,R,Lps,Rps,nv = build_slice(ss, rng, i, newv)
    # println("done! (L, R) = ($L, $R)"); flush(stdout)
    nvs     = nv + 2                     # number of V(x) evaluations = 2 (init_slice) + nv (grow_slice)
    # println("\tshrinking slice..."); flush(stdout)
    newxi,newps,nv = shrink_slice(ss, rng, i, L, R, Lps, Rps, newv)
    # println("\tdone! newxi=$newxi"); flush(stdout)
    nvs    += nv
    # print("\tsetting new state..."); flush(stdout)
    ss.x[i] = newxi                      # update state
    # print("done!\n\tsetting new potentials..."); flush(stdout)
    set_potentials!(ss, newps...)        # update potentials
    # println("done!\nFinished dimension i=$i."); flush(stdout)
    return nvs
end

# multivariate step: Gibbs update
function step!(ss::SliceSampler, rng::AbstractRNG)
    nvs = zero(ss.p)                     # number of V(x) evaluations
    for i in eachindex(ss.x)
        # println("step!: sampling dimension i=$i"); flush(stdout)
        nvs += step!(ss, rng, i)
    end
    return one(eltype(ss.curV)), nvs     # return "acceptance probability" and number of V evals
end
