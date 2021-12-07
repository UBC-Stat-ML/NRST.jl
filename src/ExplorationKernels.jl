# abstract type for Exploration kernels
# must have fields
#   U: energy function of distribution targeted by the kernel
# must implement methods
#   set_state!:
#   explore!:
#   tune!:
abstract type ExplorationKernel end

###############################################################################
# IID sampling
###############################################################################

# note that the struct contains the user-supplied energy function
# this is similar to the way user-supplied functions are handled in DifferentialEquations
# see e.g. https://github.com/SciML/SciMLBase.jl/blob/0c3ff86218c23a73389700b2f8eb057d7b95630b/src/problems/ode_problems.jl#L19
# in other words, this is not "trying to do O.O. stuff in Julia"
struct IIDSampler{F,G} <: ExplorationKernel
    U::F # energy function
    rand::G # produce one iid sample 
end
set_state!(iids::IIDSampler,extra...) = (return) # no state, nothing to do
# exploration is just iid sampling
explore!(iids::IIDSampler,extra...) = iids.rand()
# no tuning, just approximate E[V]
tune!(iids::IIDSampler,V;nsteps=50) = mean([V(iids.rand()) for _ in 1:nsteps])

# # test
# # approximate V(x)=0.5||x||^2 (energy of N(0,I)) with reference N(0,b^2I). Then
# # E[V] = (1/2)E[X1^2+X2^2] = (1/2) b^2 E[(X1/b)^2+(X2/b)^2] = b^2/2E[chi-sq(2)] = b^2
# iids=IIDSampler(x->0.125sum(abs2,x),()->2randn(2)) # b=2 => E[V]=4
# explore!(iids)
# tune!(iids,x->(0.5sum(abs2,x)),nsteps=5000)

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
    U::F # energy function = -log-density of target distribution
    x::A # current state
    xprop::A # storage for proposal
    sigma::Base.RefValue{K}  # step length (stored as ref to make it mutable)
    curU::Base.RefValue{K} # current energy (stored as ref to make it mutable) 
end

# simple outer constructor
MHSampler(U,xinit,sigma=1.0) = MHSampler(U,xinit,similar(xinit),Ref(sigma),Ref(U(xinit)))

# # test
# mhs = MHSampler(x->(0.5sum(abs2,x)),ones(2))

# set_state: change x and curU 
function set_state!(mhs::MHSampler,xinit)
    copyto!(mhs.x,xinit)
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
    return mhs.x
end

# run sampler keeping track of real-valued function V and number of accepted proposals
function run_with_trace!(mhs::MHSampler, V, tracev::AbstractVector{<:Real})
    nacc = 0
    for n in 1:length(tracev)
        step!(mhs) && (nacc += 1)
        tracev[n] = V(mhs.x)
    end
    return nacc
end

# # test
# tracev = Vector{Float64}(undef, 1000)
# nacc = run_with_trace!(mhs,x->(0.5sum(abs2,x)),tracev)

# tune sigma and approximate E[V]
function tune!(mhs::MHSampler,V; nsteps=500, target_acc=0.234, eps=0.03, max_rounds=8, verbose=false)
    meanV = 0.0; err = eps; r = 0 # initialize
    tracev = Vector{typeof(V(mhs.x))}(undef, nsteps) # allocate storage for trace of V
    while err >= eps && r < max_rounds
        r += 1
        nacc = run_with_trace!(mhs,V,tracev)
        acc = nacc/nsteps # compute acceptance ratio
        err = abs(acc-target_acc) # absolute error
        old_sigma = mhs.sigma[]
        mhs.sigma[] = old_sigma + 10(r^(-0.51))*(acc-target_acc) # simplified SGD for 0.5(acc-target)^2 with R-M seq a_r = 10r^{-0.51}
        meanV = (r==1) ? mean(tracev) : 0.5(meanV+mean(tracev)) # use geometric weighting to give more importance to rounds with better tuning
        verbose && @printf(
            "Round %d: acc=%.3f, err=%.2f, sigma=%.1f, meanV=%.1f\n",
            r,acc,err,old_sigma,meanV
        )
    end
    return meanV
end

# # test
# # approximate V(x)=0.5||x||^2 (energy of N(0,I)) with mhs targetting N(0,b^2I). Then
# # E[V] = (1/2)E[X1^2+X2^2] = (1/2) b^2 E[(X1/b)^2+(X2/b)^2] = b^2/2E[chi-sq(2)] = b^2
# mhs = MHSampler(x->(0.125sum(abs2,x)),ones(2),2.3) # b=2 => E[V]=4
# tune!(mhs,x->(0.5sum(abs2,x)),verbose=true,nsteps=5000)
# mhs.sigma[] = 4.0 # sigma too high
# @code_warntype tune!(mhs,x->(0.5sum(abs2,x)))

###############################################################################
# construction of vectors of exploration kernels
# these are usually mixed, with 1st element an IIDSampler and the rest an MCMC
# sampler
###############################################################################

# union of concrete types
# useful for having vectors of mixed types (as long as <=4 types)
# see https://stackoverflow.com/a/58539098/5443023
# also https://docs.julialang.org/en/v1/manual/types/#citeref-1
const ContinuousSampler = Union{IIDSampler, MHSampler}

# create a vector of exploration kernels: continous case
function init_explorers(V,Vref,randref,betas,xinit::AbstractVector{<:AbstractFloat})
    A = Vector{ContinuousSampler}(undef, length(betas))
    A[1] = IIDSampler(Vref,randref)
    for i in 2:length(betas)
        beta = betas[i] # better to extract the beta, o.w. the closure grabs the whole vector
        A[i] = MHSampler(x->(Vref(x) + beta*V(x)),xinit)
    end
    return A
end