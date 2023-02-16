###############################################################################
# Metropolis-Hastings sampler with isotropic proposal
# TODO: the proposal (and its tuning) is the only thing not reusable for other
# statespaces and symmetric proposals! We must
# - make MHSampler parametrized on type of x::T, where T can be in principle anything 
# - put x::T and xprop::T
# - replace sigma with something more generic like "params" 
# - implement different propose! that dispatch on different types of x's
###############################################################################

struct MHSampler{TTM<:TemperedModel,K<:AbstractFloat,A<:AbstractVector{K}} <: ExplorationKernel
    # fields common to every ExplorationKernel
    tm::TTM                   # TemperedModel
    x::A                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{K}    # current beta
    curVref::Base.RefValue{K} # current reference potential
    curV::Base.RefValue{K}    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{K}   # current tempered potential
    # idiosyncratic fields
    xprop::A                  # storage for proposal
    sigma::Base.RefValue{K}   # step length (stored as ref to make it mutable) 
end

# outer constructor
function MHSampler(tm, x, curβ, curVref, curV, curVβ; sigma=1.0)
    MHSampler(tm, x, curβ, curVref, curV, curVβ, similar(x), Ref(sigma))
end
params(mh::MHSampler) = (sigma=mh.sigma[],)                   # get params as namedtuple
function set_params!(mh::MHSampler, params::NamedTuple)       # set sigma from a NamedTuple
    mh.sigma[] = params.sigma
end

# copy constructor
function Base.copy(mh::MHSampler, args...)
    MHSampler(args..., similar(mh.xprop), Ref(mh.sigma[]))
end

# MH proposal = isotropic normal
function propose!(mhs::MHSampler, rng::AbstractRNG)
    for (i,xi) in enumerate(mhs.x)
        @inbounds mhs.xprop[i] = xi + mhs.sigma[]*randn(rng)
    end
end

# accept proposal: x <- xprop and update potentials 
function acc_prop!(mhs::MHSampler, vref::F, v::F, vβ::F) where {F<:AbstractFloat}
    set_x!(mhs, mhs.xprop)
    set_potentials!(mhs, vref, v, vβ)
end

# basic Metropolis step (i.e, with symmetric proposal)
# note: acc is sampled with the exponential r.v. approach
# u < pi(q)/pi(p) <=> log(pi(u)) < log(pi(q)) -log(pi(p)) 
# <=> Exp(1)=(dist) -log(u) > (-log(pi(q))) - (-log(pi(p))) = U(q) - U(p) = ΔU
# note: to get back the acceptance ratio
# pi(q)/pi(p) = exp(log(pi(q)) - log(pi(p))) = exp(-[U(q) - U(p)]) = exp(-ΔU)
# => min(1,pi(q)/pi(p)) = min(1,exp(-ΔU)) = exp[log(min(1,exp(-ΔU)))] 
# = exp[min(0,-ΔU)] = exp[-max(0,ΔU)]
function step!(mhs::MHSampler, rng::AbstractRNG)
    @unpack xprop = mhs
    propose!(mhs, rng)                      # propose new state, stored at mhs.xprop
    pvref, pv, pvβ = potentials(mhs, xprop) # compute potentials at proposed location
    ΔU  = pvβ - mhs.curVβ[]                 # compute energy differential
    acc = ΔU < randexp(rng)                 # twice as fast than -log(rand())
    acc && acc_prop!(mhs, pvref, pv, pvβ)   # if proposal accepted, update state and caches
    ap  = exp(-max(zero(ΔU), ΔU))           # compute acceptance probability
    return ap, 1                            # return ap and 1 == number of V(x) evals -> happens inside potentials()
end

# run sampler keeping track only of cummulative acceptance probability
# used in tuning
function run!(mhs::MHSampler, rng::AbstractRNG, nsteps::Int)
    sum_ap = 0.
    for _ in 1:nsteps
        ap, _   = step!(mhs, rng)
        sum_ap += ap
    end
    return (sum_ap/nsteps)
end

# run sampler keeping track of V
function run!(mhs::MHSampler{F,K}, rng::AbstractRNG, trV::Vector{K}) where {F,K}
    sum_ap = zero(K)
    nsteps = length(trV)
    for n in 1:nsteps
        ap, _   = step!(mhs, rng)
        sum_ap += ap
        trV[n]  = mhs.curV[]
    end
    return (sum_ap/nsteps)
end
# # test
# tracev = Vector{Float64}(undef, 1000)
# nacc = run!(mhs,x->(0.5sum(abs2,x)),tracev)

# tune sigma using a grid search approach
function tune!(
    mhs::MHSampler{F,K},
    rng::AbstractRNG;
    target_acc = 0.234,
    nsteps     = 512,      # for reference: sample size required computed via p(1-p)(Z/eps)^2 <= (Z/2eps)^2, for Z = quantile(Normal(), 0.975)
    erange     = (-3.,3.), # range for the exponent 
    tol        = 0.05
    ) where {F,K}
    # find e so that sigma=oldsigma*10^e produces the acc rate most similar to target
    # true acc(e) must be monotone in theory. noisy version may not but :shrug:
    oldsigma = mhs.sigma[]
    function tfun(e)
        mhs.sigma[] = oldsigma * (10^e)
        run!(mhs, rng, nsteps) - target_acc
    end
    eopt, fopt  = monoroot(tfun, erange...; tol = tol, maxit = 8)
    mhs.sigma[] = oldsigma * (10^eopt)
    # @debug "tune-xpl: setting σ=$(mhs.sigma[]) with acc=$(fopt+target_acc)"
    any(erange .≈ eopt) && 
        @warn "eopt=$eopt is a boundary of range=$erange. Got: oldsigma" *
              "=$oldsigma, sigma=$(mhs.sigma[]), acc=$(fopt+target_acc)."
    return
end

# smooth with running median
# intuition: if optimal sigma=sigma(β) is smooth in β, then tuning can be improved
# by smoothing across explorers
# Note: instead of working in x = betas scale, we use x = "equal units of rejection"
# assuming perfect tuning.
function smooth_params!(xpls::Vector{<:MHSampler}, λ::AbstractFloat)
    σs  = [first(params(xpl)) for xpl in xpls]
    w   = closest_odd(λ*length(xpls)) 
    pσs = running_median(σs, w, :asymmetric_truncated) # asymmetric_truncated also smooths endpoints 
    for (i,xpl) in enumerate(xpls)
        xpl.sigma[] = pσs[i]
    end
end


###############################################################################
# set MHSampler as default kernel for some statespaces
###############################################################################

# default exploration kernel for x in ℝᵈ
function get_explorer(
    tm::TemperedModel, 
    xinit::AbstractVector{TF}, 
    refV::Base.RefValue{TF}
    ) where {TF<:AbstractFloat}
    # note: by not copying xinit, NRSTSampler and explorer share the same x state
    #       same with the cached potential V 
    # note: tm might contain fields that are mutated whenever potentials are
    # computed (e.g. for TuringModel)
    MHSampler(tm, xinit, one(TF), refV)
end


