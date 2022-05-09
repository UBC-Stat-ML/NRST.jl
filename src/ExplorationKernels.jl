###############################################################################
# exploration kernels
###############################################################################

abstract type ExplorationKernel end

# explore == run for nsteps
function explore!(ex::ExplorationKernel, nsteps::Int)
    for _ in 1:nsteps
        step!(ex)
    end
end

# change x and update potentials 
function set_state!(ex::ExplorationKernel, x, vref, v, vβ)
    set_x!(ex, x)
    set_potentials!(ex, vref, v, vβ)
end
function set_state!(ex::ExplorationKernel, x, v)
    vref = Vref(ex.tm, x)
    vβ   = vref + ex.betas[ex.id]*v
    set_state!(ex, x, vref, v, vβ)
end
function set_potentials!(ex::ExplorationKernel, vref, v, vβ)
    ex.curVref[] = vref
    ex.curV[]    = v
    ex.curVβ[]   = vβ
end

# compute potentials at some x
function potentials(ex::ExplorationKernel, newx)
    vref, v = potentials(ex.tm, newx)
    vβ = vref + ex.betas[ex.id]*v
    vref, v, vβ
end

# called by functions that modify "betas" during tuning
function refresh_curVβ!(ex::ExplorationKernel)
    ex.curVβ[] = ex.curVref[] + ex.betas[ex.id]*ex.curV[]
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

struct MHSampler{TTM<:TemperedModel,K<:AbstractFloat,A<:AbstractVector{K},I<:Int} <: ExplorationKernel
    tm::TTM                   # TemperedModel
    x::A                      # current state
    xprop::A                  # storage for proposal
    sigma::Base.RefValue{K}   # step length (stored as ref to make it mutable)
    betas::A                  # pointer to the (unique) NRST betas vector
    id::I                     # id of explorer == position of the β in betas that it uses. note: id>=2, because β=0 does not require an explorer 
    curVref::Base.RefValue{K} # current reference potential
    curV::Base.RefValue{K}    # current target potential
    curVβ::Base.RefValue{K}   # current tempered potential 
end

# outer constructor
function MHSampler(tm, xinit, betas, id, sigma=1.0)
    vref, v = potentials(tm, xinit)
    vβ = vref + betas[id]*v
    MHSampler(
        tm,
        xinit,
        similar(xinit),
        Ref(sigma),
        betas,
        id,
        Ref(vref),
        Ref(v),
        Ref(vβ)
    )
end
params(mh::MHSampler) = (sigma=mh.sigma[],) # namedtuple of parameters
set_x!(mhs, newx) = copyto!(mhs.x, newx)    # set x to new value
# # test
# mhs = MHSampler(x->(0.5sum(abs2,x)),ones(2))


# set_state!(mhs,[0.2,0.5]) # test

# MH proposal = isotropic normal
function propose!(mhs::MHSampler)
    for (i,xi) in enumerate(mhs.x)
        mhs.xprop[i] = xi + mhs.sigma[]*randn()
    end
end

# basic Metropolis step (i.e, with symmetric proposal)
# note: acc is sampled with the exponential r.v. approach
# u < pi(q)/pi(p) <=> log(pi(u)) < log(pi(q)) -log(pi(p)) 
# <=> Exp(1)=(dist) -log(u) > (-log(pi(q))) - (-log(pi(p))) = U(q) - U(p) = ΔU
# note: to get back the acceptance ratio
# pi(q)/pi(p) = exp(log(pi(q)) - log(pi(p))) = exp(-[U(q) - U(p)]) = exp(-ΔU)
# => min(1,pi(q)/pi(p)) = min(1,exp(-ΔU)) = exp[log(min(1,exp(-ΔU)))] 
# = exp[min(0,-ΔU)] = exp[-max(0,ΔU)]
function step!(mhs::MHSampler)
    @unpack xprop = mhs
    propose!(mhs)                             # propose new state, stored at mhs.xprop
    pvref, pv, pvβ = potentials(mhs, xprop)   # compute potentials at proposed location
    ΔU  = pvβ - mhs.curVβ[]                   # compute energy differential
    acc = ΔU < rand(Exponential())            # twice as fast than -log(rand())
    acc && set_state!(mhs,xprop,pvref,pv,pvβ) # if proposal accepted, update state
    ap  = exp(-max(0., ΔU))                   # compute acceptance probability
    return ap
end

# # test
# println(step!(mhs))
# println(mhs.curU[])
# mhs.x

# run sampler keeping track of accepted probabilities
function run!(mhs::MHSampler, nsteps::Int)
    sum_ap = 0.
    for n in 1:nsteps
        sum_ap += step!(mhs)
    end
    return (sum_ap/nsteps)
end

# run sampler keeping track of V
function run!(mhs::MHSampler{F,K}, trV::AbstractVector{K}) where {F,K}
    sum_ap = zero(K)
    nsteps = length(trV)
    for n in 1:nsteps
        sum_ap += step!(mhs)
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
    mhs::MHSampler{F,K};
    sigma0     = -one(K),
    target_acc = 0.234,
    eps        = 0.05,
    α          = -1.0,
    min_rounds = 2,
    max_rounds = 16,
    nsteps     = 400,
    verbose    = true
    ) where {F,K}
    nsteps < 1 && return
    minsigma = 1e-8
    err      = 10*eps
    r        = 0
    (sigma0 > zero(K)) && (mhs.sigma[] = sigma0)
    verbose && @printf("Tuning initiated at sigma=%.1f\n", mhs.sigma[])
    while (r < min_rounds) || (err >= eps && r < max_rounds)
        r += 1
        acc          = run!(mhs, nsteps)          # run and get average acceptance probability
        err          = abs(acc - target_acc)      # absolute error
        mhs.sigma[] += 10(r^α)*(acc - target_acc) # SGD step. "10" should work for most settings since scale is acc ratio, which is universal 
        mhs.sigma[]  = max(minsigma, mhs.sigma[]) # project back to >0
        verbose && @printf(
            "Round %d: acc=%.3f, err=%.2f, new_sigma=%.1f\n",r, acc, err, mhs.sigma[]
        )
    end
end
tune!(mhs::MHSampler,params::NamedTuple;kwargs...) = tune!(mhs::MHSampler;sigma0=params.sigma,kwargs...)
# # test
# # approximate V(x)=0.5||x||^2 (energy of N(0,I)) with mhs targetting N(0,b^2I). Then
# # E[V] = (1/2)E[X1^2+X2^2] = (1/2) b^2 E[(X1/b)^2+(X2/b)^2] = b^2/2E[chi-sq(2)] = b^2
# mhs = MHSampler(x->(0.125sum(abs2,x)),ones(2),2.3) # b=2 => E[V]=4
# tune!(mhs,x->(0.5sum(abs2,x)),verbose=true,nsteps=5000)
# mhs.sigma[] = 4.0 # sigma too high
# @code_warntype tune!(mhs,x->(0.5sum(abs2,x)))

###############################################################################
# methods for vectors of exploration kernels
###############################################################################

# create a vector of exploration kernels
function init_explorers(tm::TemperedModel, betas, xinit::AbstractVector{<:AbstractFloat})
    # copy(xinit) is necessary or all explorers end up sharing the same state x
    # copy(tm) is necessary for threading when evaluating V,Vref,randref 
    # changes something in their closures (e.g., when tm isa TuringTemperedModel)
    # but "betas" is shared because for exploration this is read-only.
    exp1    = MHSampler(copy(tm), copy(xinit), betas, 2)
    exps    = Vector{typeof(exp1)}(undef, length(betas) - 1)
    exps[1] = exp1
    for i in 3:length(betas)
        exps[i-1] = MHSampler(copy(tm), copy(xinit), betas, i)
    end
    return exps
end

# intuition: if optimal sigma=sigma(β) is smooth in β, then tuning can be improved
# by smoothing across explorers
function smooth_params!(explorers::Vector{<:MHSampler})
    betas     = explorers[1].betas
    logσs     = [log(NRST.params(e)[1]) for e in explorers]
    spl       = fit(SmoothingSpline, log.(betas[2:end]), logσs, 20.)
    logσspred = predict(spl)
    for (i,e) in enumerate(explorers)
        e.sigma[] = exp(logσspred[i])
    end
end
