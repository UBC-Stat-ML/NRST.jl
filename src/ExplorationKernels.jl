# abstract type for Exploration kernels
# must have fields
#   U: energy function of distribution targeted by the kernel
# must implement methods
#   set_state!:
#   explore!:
#   tune!:
abstract type ExplorationKernel end

###############################################################################
# Metropolis-Hastings sampler with isotropic proposal
# TODO: the proposal (and its tuning) is the only thing not reusable for other
# targets! We must
# - make MHSampler parametrized on type of x::T, where T can be in principle anything 
# - put x::T and xprop::T
# - replace sigma with something more generic like "params" 
# - implement different propose! that dispatch on different types of x's
###############################################################################

struct MHSampler{F,K<:AbstractFloat,A<:AbstractVector{K}} <: ExplorationKernel
    U::F                    # energy function = -log-density of target distribution
    x::A                    # current state
    xprop::A                # storage for proposal
    sigma::Base.RefValue{K} # step length (stored as ref to make it mutable)
    curU::Base.RefValue{K}  # current energy (stored as ref to make it mutable) 
end

MHSampler(U,xinit,sigma=1.0) = MHSampler(U,xinit,similar(xinit),Ref(sigma),Ref(U(xinit)))
params(mh::MHSampler) = (sigma=mh.sigma[],) # namedtuple of parameters

# # test
# mhs = MHSampler(x->(0.5sum(abs2,x)),ones(2))

# set_state: change x and curU 
function set_state!(mhs::MHSampler,x)
    copyto!(mhs.x, x)
    mhs.curU[] = mhs.U(x)
end

# set_state!(mhs,[0.2,0.5]) # test

# MH proposal = isotropic normal
function propose!(mhs::MHSampler)
    @unpack x,xprop,sigma = mhs
    s = sigma[]
    for i in eachindex(xprop)
        xprop[i] = x[i] + s*randn()
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
    @unpack U,xprop,curU = mhs
    propose!(mhs)                    # propose new state, stored at mhs.xprop
    propU = U(xprop)                 # compute energy at proposed location
    ΔU    = propU - curU[]
    acc   = ΔU < rand(Exponential()) # twice as fast than -log(rand())
    if acc
        curU[] = propU               # update current energy if accepted
        copyto!(mhs.x,mhs.xprop)     # update state
    end
    ap    = exp(-max(0., ΔU))        # acceptance probability
    return ap
end

# # test
# println(step!(mhs))
# println(mhs.curU[])
# mhs.x

# for MHSampler, explore == run for nsteps
function explore!(mhs::MHSampler, nsteps::Int)
    for n in 1:nsteps
        step!(mhs)
    end
end

# run sampler keeping track of accepted probabilities
function run!(mhs::MHSampler, nsteps::Int)
    sum_ap = 0.
    for n in 1:nsteps
        sum_ap += step!(mhs)
    end
    return (sum_ap/nsteps)
end

# run sampler keeping track of real-valued function V and number of accepted proposals
function run!(
    mhs::MHSampler{F,K},
    V,
    tracev::AbstractVector{K}
    ) where {F,K}
    sum_ap = 0.
    nsteps = length(tracev)
    for n in 1:nsteps
        sum_ap   += step!(mhs)
        tracev[n] = V(mhs.x)
    end
    return (sum_ap/nsteps)
end
# # test
# tracev = Vector{Float64}(undef, 1000)
# nacc = run!(mhs,x->(0.5sum(abs2,x)),tracev)

# tune sigma using a simplified SGD approach targetting
# 0.5(acc-target)^2, where 
#     - \der{acc}{sigma} is assumed constant
#     - Robbins-Monroe step size sequence is a_r = 10r^α for α<0
function tune!(
    mhs::MHSampler{F,K};
    sigma0     = -one(K),
    target_acc = 0.234,
    eps        = 0.03,
    α          = -1.0,
    min_rounds = 2,
    max_rounds = 16,
    nsteps     = 500,
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
# construction of vectors of exploration kernels
###############################################################################

# create a vector of exploration kernels: continous case
function init_explorers(fns::Funs,betas,xinit::AbstractVector{<:AbstractFloat})
    # copy(xinit) is necessary or all explorers end up sharing the same state x
    [MHSampler(gen_Vβ(fns, i, betas), copy(xinit)) for i in 2:length(betas)]
end

