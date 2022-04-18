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
function set_state!(mhs::MHSampler,xinit)
    copyto!(mhs.x, xinit)
    mhs.curU[] = mhs.U(xinit)
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

# u < pi(q)/pi(p) <=> log(pi(u)) < log(pi(q)) -log(pi(p)) 
# <=> Exp(1)=(dist) -log(u) > (-log(pi(q))) - (-log(pi(p))) = U(q) - U(p)
function accrej(mhs::MHSampler)
    @unpack U,xprop,curU = mhs
    propU = U(xprop) # compute energy at proposed location
    acc = (propU - curU[]) < rand(Exponential()) # twice as fast than -log(rand())
    acc && (curU[] = propU) # update current energy if accepted
    return acc
end

function step!(mhs::MHSampler)
    propose!(mhs)
    acc = accrej(mhs)
    acc && copyto!(mhs.x,mhs.xprop)
    return acc
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

# run sampler keeping track of accepted proposals
function run!(mhs::MHSampler, nsteps::Int)
    nacc = 0
    for n in 1:nsteps
        step!(mhs) && (nacc += 1)
    end
    return nacc
end

# run sampler keeping track of real-valued function V and number of accepted proposals
function run!(
    mhs::MHSampler{F,K},
    V,
    tracev::AbstractVector{K}
    ) where {F,K}
    nacc = 0
    for n in 1:length(tracev)
        step!(mhs) && (nacc += 1)
        tracev[n] = V(mhs.x)
    end
    return nacc
end
# # test
# tracev = Vector{Float64}(undef, 1000)
# nacc = run!(mhs,x->(0.5sum(abs2,x)),tracev)

# tune sigma using a simplified SGD approach targetting
# 0.5(acc-target)^2, where 
#     - \der{acc}{sigma} is assumed constant
#     - Robbins-Monroe step size sequence is a_r = 10r^{-0.51}
function tune!(
    mhs::MHSampler{F,K};
    sigma0     = -one(K),
    target_acc = 0.234,
    eps        = 0.03,
    min_rounds = 2,
    max_rounds = 8,
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
        nacc         = run!(mhs, nsteps)                # run and get number of acc proposals
        acc          = nacc / nsteps                    # compute acceptance ratio
        err          = abs(acc - target_acc)            # absolute error
        mhs.sigma[] += 10(r^(-0.51))*(acc - target_acc) # SGD step. "10" should work for most settings since scale is acc ratio, which is universal 
        mhs.sigma[]  = max(minsigma, mhs.sigma[])       # project back to >0
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

