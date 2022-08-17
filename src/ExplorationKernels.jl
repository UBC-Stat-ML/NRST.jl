###############################################################################
# exploration kernels
###############################################################################

abstract type ExplorationKernel end

#######################################
# methods interfacing with an NRSTSampler
#######################################

# explore for nsteps using the given params, without keeping track of anything
# note: this only makes sense when called by an NRSTSampler with which the
# explorer shares x and curV
function explore!(
    ex::ExplorationKernel,
    rng::AbstractRNG,
    β::AbstractFloat,
    params::NamedTuple,
    nsteps::Int
    )
    update_β!(ex, β)
    set_params!(ex, params)
    acc = zero(typeof(β))
    for _ in 1:nsteps
        acc += step!(ex, rng)
    end
    return acc/nsteps       # return average acceptance probability
end
function update_β!(ex::ExplorationKernel, β::AbstractFloat)
    ex.curβ[]    = β
    ex.curVref[] = Vref(ex.tm, ex.x)          # vref is *not* shared so needs updating
    ex.curVβ[]   = ex.curVref[] + β*ex.curV[] # vβ is *not* shared so needs updating
end

#######################################
# methods used only within an explorer
#######################################

# compute all potentials at some x
function potentials(ex::ExplorationKernel, newx)
    vref, v = potentials(ex.tm, newx)
    vβ = vref + ex.curβ[]*v
    vref, v, vβ
end

# set potentials. used during sampling with an explorer
function set_potentials!(ex::ExplorationKernel, vref::F, v::F, vβ::F) where {F<:AbstractFloat}
    ex.curVref[] = vref
    ex.curV[]    = v
    ex.curVβ[]   = vβ
end

###############################################################################
# Metropolis-Hastings sampler with isotropic proposal
# TODO: the proposal (and its tuning) is the only thing not reusable for other
# targets! We must
# - make MHSampler parametrized on type of x::T, where T can be in principle anything 
# - put x::T and xprop::T
# - replace sigma with something more generic like "params" 
# - implement different propose! that dispatch on different types of x's
###############################################################################

struct MHSampler{TTM<:TemperedModel,K<:AbstractFloat,A<:AbstractVector{K}} <: ExplorationKernel
    tm::TTM                   # TemperedModel
    x::A                      # current state
    xprop::A                  # storage for proposal
    sigma::Base.RefValue{K}   # step length (stored as ref to make it mutable)
    curβ::Base.RefValue{K}    # current beta
    curVref::Base.RefValue{K} # current reference potential
    curV::Base.RefValue{K}    # current target potential
    curVβ::Base.RefValue{K}   # current tempered potential 
end

# outer constructor
function MHSampler(tm, xinit, β, curV, sigma=1.0)
    vref    = Vref(tm, xinit)
    vβ      = vref + β*curV[]
    MHSampler(
        tm, xinit, similar(xinit), Ref(sigma), Ref(β), Ref(vref), curV, Ref(vβ)
    )
end
set_x!(mhs, newx) = copyto!(mhs.x, newx)                # set x to new value
params(mh::MHSampler) = (sigma=mh.sigma[],)             # namedtuple of parameters
function set_params!(mh::MHSampler, params::NamedTuple) # set sigma from a NamedTuple
    mh.sigma[] = params.sigma
end

# copy constructors
function Base.copy(mh::MHSampler)
    newtm = copy(mh.tm)
    newx  = copy(mh.x)
    ncurV = Ref(mh.curV[])
    copy(mh, newtm, newx, ncurV)
end
function Base.copy(mh::MHSampler, newtm::TemperedModel, newx, newcurV)
    MHSampler(
        newtm, newx, similar(mh.xprop), Ref(mh.sigma[]), Ref(mh.curβ[]),
        Ref(mh.curVref[]), newcurV, Ref(mh.curVβ[])
    )
end

# MH proposal = isotropic normal
function propose!(mhs::MHSampler, rng::AbstractRNG)
    for (i,xi) in enumerate(mhs.x)
        mhs.xprop[i] = xi + mhs.sigma[]*randn(rng)
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
    ap  = exp(-max(0., ΔU))                 # compute acceptance probability
    return ap
end

# run sampler keeping track only of cummulative acceptance probability
# used in tuning
function run!(mhs::MHSampler, rng::AbstractRNG, nsteps::Int)
    sum_ap = 0.
    for n in 1:nsteps
        sum_ap += step!(mhs, rng)
    end
    return (sum_ap/nsteps)
end

# run sampler keeping track of V
function run!(
    mhs::MHSampler{F,K},
    rng::AbstractRNG,
    trV::AbstractVector{K}
    ) where {F,K}
    sum_ap = zero(K)
    nsteps = length(trV)
    for n in 1:nsteps
        sum_ap += step!(mhs, rng)
        trV[n]  = mhs.curV[]
    end
    return (sum_ap/nsteps)
end
# # test
# tracev = Vector{Float64}(undef, 1000)
# nacc = run!(mhs,x->(0.5sum(abs2,x)),tracev)

# tune sigma using a simplified SGD approach targeting 0.5(acc-target)^2, where 
# - \der{acc}{sigma} is assumed constant
# - Robbins-Monroe step size sequence is a_r = 10r^α for α<0
function tune!(
    mhs::MHSampler{F,K},
    rng::AbstractRNG;
    target_acc = 0.234,
    eps        = 0.05,
    α          = -1.0,
    min_rounds = 2,
    max_rounds = 16,
    nsteps     = 400, # ~= p(1-p)(Z/eps)^2 <= (Z/2eps)^2, for Z = quantile(Normal(), 0.975)
    verbose    = true
    ) where {F,K}
    nsteps < 1 && return
    minsigma = 1e-8
    err      = 10*eps
    r        = 0
    verbose && @printf("Tuning initiated at sigma=%.1f\n", mhs.sigma[])
    while (r < min_rounds) || (err >= eps && r < max_rounds)
        r += 1
        acc          = run!(mhs, rng, nsteps)     # run and get average acceptance probability
        err          = abs(acc - target_acc)      # absolute error
        mhs.sigma[] += 10(r^α)*(acc - target_acc) # SGD step. "10" should work for most settings since scale is acc ratio, which is universal 
        mhs.sigma[]  = max(minsigma, mhs.sigma[]) # project back to >0
        verbose && @printf(
            "Round %d: acc=%.3f, err=%.2f, new_sigma=%.1f\n",r, acc, err, mhs.sigma[]
        )
    end
end
# # test
# # approximate V(x)=0.5||x||^2 (energy of N(0,I)) with mhs targetting N(0,b^2I). Then
# # E[V] = (1/2)E[X1^2+X2^2] = (1/2) b^2 E[(X1/b)^2+(X2/b)^2] = b^2/2E[chi-sq(2)] = b^2
# mhs = MHSampler(x->(0.125sum(abs2,x)),ones(2),2.3) # b=2 => E[V]=4
# tune!(mhs,x->(0.5sum(abs2,x)),verbose=true,nsteps=5000)
# mhs.sigma[] = 4.0 # sigma too high
# @code_warntype tune!(mhs,x->(0.5sum(abs2,x)))

###############################################################################
# default kernels for given data types
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
    MHSampler(tm, xinit, one(first(xinit)), refV)
end

###############################################################################
# methods for collections of exploration kernels
###############################################################################

# instantiate a vector of explorers by copying one. used for tuning NRSTSampler
# note: the resulting explorers are not tethered to any NRSTSampler, in the sense 
# that the state x is not shared with any NRSTSampler. Conversely, if xpl is
# the explorer of an NRSTSampler, this function does not change that.
function replicate(xpl::ExplorationKernel, betas::AbstractVector{<:AbstractFloat})
    N    = length(betas) - 1
    xpls = Vector{typeof(xpl)}(undef, N)
    for i in 1:N
        newxpl  = copy(xpl)           # use custom copy constructor
        update_β!(newxpl, betas[i+1]) # set β and recalculate Vβ
        xpls[i] = newxpl
    end
    return xpls
end

# intuition: if optimal sigma=sigma(β) is smooth in β, then tuning can be improved
# by smoothing across explorers
function smooth_params!(xpls::Vector{<:MHSampler})
    betas     = [xpl.curβ[] for xpl in xpls] # note: this is length N
    logσs     = [log(params(xpl)[1]) for xpl in xpls]
    spl       = fit(SmoothingSpline, log.(betas), logσs, 20.)
    logσspred = predict(spl)
    for (i,xpl) in enumerate(xpls)
        xpl.sigma[] = exp(logσspred[i])
    end
end

