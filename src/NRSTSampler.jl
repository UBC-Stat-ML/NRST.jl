###############################################################################
# relevant structs 
###############################################################################

# encapsulates all the specifics of the tempered problem
struct NRSTProblem{TTM<:TemperedModel,K<:AbstractFloat,A<:Vector{K},TInt<:Int,TNT<:NamedTuple}
    tm::TTM              # a TemperedModel
    N::TInt              # number of states additional to reference (N+1 in total)
    betas::A             # vector of tempering parameters (length N+1)
    c::A                 # vector of parameters for the pseudoprior
    use_mean::Bool       # should we use "mean" (true) or "median" (false) for tuning c?
    nexpls::Vector{TInt} # vector of length N with number of exploration steps adequate for each level 1:N
    xplpars::Vector{TNT} # vector of length N of named tuples, holding adequate parameters to use at each level 1:N
end

# copy constructor, allows replacing tm, but keeps everything else
function NRSTProblem(oldnp::NRSTProblem, newtm)
    NRSTProblem(
        newtm,oldnp.N,oldnp.betas,oldnp.c,oldnp.use_mean,oldnp.nexpls,oldnp.xplpars
    )
end

# struct for the sampler
struct NRSTSampler{T,I<:Int,K<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

# raw trace of a serial run
struct SerialNRSTTrace{T,TI<:Int,TF<:AbstractFloat}
    trX::Vector{T}
    trIP::Vector{SVector{2,TI}}
    trV::Vector{TF}
    trRP::Vector{TF}
    N::TI
end
nsteps(tr::SerialNRSTTrace) = length(tr.trV) # recover nsteps

# processed output
struct SerialRunResults{T,TI<:Int,TF<:AbstractFloat} <: RunResults
    tr::SerialNRSTTrace{T,TI} # raw trace
    xarray::Vector{Vector{T}} # i-th entry has samples at state i
    trVs::Vector{Vector{TF}}  # i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}        # total number of visits to each (i,eps)
    rpacc::Matrix{TF}         # accumulates rejection probs of swaps started from each (i,eps)
end

###############################################################################
# constructors and initialization methods
###############################################################################

# constructor that also builds an NRSTProblem and does initial tuning
function NRSTSampler(
    tm::TemperedModel;
    betas          = nothing,
    N::Int         = 3, # best to use N(Λ) = inf{n: Λ/n ≤ 0.5}
    nexpl::Int     = 50, 
    use_mean::Bool = true,
    tune::Bool     = true,
    verbose::Bool  = false,
    kwargs...
    )
    if isnothing(betas)
        betas = init_grid(N)
    else
        N = length(betas) - 1
    end
    x    = initx(rand(tm))                                      # draw an initial point
    curV = Ref(V(tm, x))
    xpl  = get_explorer(tm, x, curV)
    np   = NRSTProblem(                                         # instantiate an NRSTProblem
        tm, N, betas, similar(betas), use_mean, fill(nexpl,N), 
        fill(params(xpl), N)
    ) 
    ns  = NRSTSampler(np, xpl, x, MVector(0,1), curV)  # instantiate the NRSTSampler
    if tune
        tunestats = tune!(ns; verbose = verbose, kwargs...)     # tune explorers, c, and betas
    else
        tunestats = NamedTuple()
    end
    return ns, tunestats
end

# grid initialization
init_grid(N::Int) = collect(range(0,1,N+1))

# safe initialization for arrays with entries in reals
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TR}) where {TR<:Real}
    uno = one(TR)
    rand(Uniform(-uno,uno), size(pre_x))
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    tm = SimpleTemperedModel(V, Vref, randref)
    NRSTSampler(tm,args...;kwargs...)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
function Base.copy(ns::NRSTSampler)
    newtm = copy(ns.np.tm)                   # the only element in np that we (may) need to copy
    newnp = NRSTProblem(ns.np, newtm)        # build new Problem using new tm
    newx  = copy(ns.x)
    ncurV = Ref(V(newtm, newx))
    nuxpl = copy(ns.xpl, newtm, newx, ncurV) # copy ns.xpl sharing stuff with the new sampler
    NRSTSampler(newnp, nuxpl, newx, MVector(0,1), ncurV)
end

###############################################################################
# sampling methods
###############################################################################

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler{T,I}) where {T,I}
    copyto!(ns.x, rand(ns.np.tm))
    ns.ip[1]  = zero(I)
    ns.ip[2]  = one(I)
    ns.curV[] = V(ns.np.tm, ns.x)
end

# communication step: the acceptance ratio is given by
# A = [pi^{(i+eps)}(x)/pi^{(i)}(x)] [p_{i+eps}/p_i]
# = [Z(i)/Z(i+eps)][exp{-b_{i+eps}V(x)}exp{b_{i}V(x)}][Z(i+eps)/Z(i)][exp{c_{i+eps}exp{-c_{i}}]
# = exp{-[b_{i+eps} - b_{i}]V(x) + c_{i+eps} -c_{i}}
# = exp{ -( [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i}) ) }
# = exp(-nlaccr)
# where nlaccr := -log(A) = [b_{i+eps} - b_{i}]V(x) - (c_{i+eps} -c_{i})
# To get the rejection probability from nlaccr:
# ap = min(1.,A) = min(1.,exp(-nlaccr)) = exp(min(0.,-nlaccr)) = exp(-max(0.,nlaccr))
# => rp = 1-ap = 1-exp(-max(0.,nlaccr)) = -[exp(-max(0.,nlaccr))-1] = -expm1(-max(0.,nlaccr))
function comm_step!(ns::NRSTSampler{T,I,K}) where {T,I,K}
    @unpack np,ip,curV = ns
    @unpack N,betas,c = np
    iprop = sum(ip)                             # propose i + eps
    if iprop < 0                                # bounce below
        ip[1] = zero(I)
        ip[2] = one(I)
        return one(K)
    elseif iprop > N                            # bounce above
        ip[1] = N
        ip[2] = -one(I)
        return one(K)
    else
        i      = ip[1]                          # current index
        nlaccr = (betas[iprop+1]-betas[i+1])*curV[] - (c[iprop+1]-c[i+1])
        acc    = (nlaccr < rand(Exponential())) # accept? Note: U<A <=> -log(A) > -log(U) ~ Exp(1) 
        if acc
            ip[1] = iprop                       # move
        else
            ip[2] = -ip[2]                      # flip direction
        end
    end
    rp = -expm1(-max(zero(K), nlaccr))
    return rp
end

# exploration step
function expl_step!(ns::NRSTSampler{T,I}) where {T,I}
    @unpack np,xpl,ip,curV = ns
    if ip[1] == zero(I)
        copyto!(ns.x, rand(np.tm))      # sample new state from the reference
        curV[] = V(np.tm, ns.x)         # compute energy at new point
    else
        β      = np.betas[ip[1]+1]      # get the β for the level
        nexpl  = np.nexpls[ip[1]]       # get number of exploration steps needed at this level
        params = np.xplpars[ip[1]]      # get explorer params for this level 
        explore!(xpl, β, params, nexpl) # explore for nexpl steps. note: ns.x and ns.curV are shared with xpl
    end
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler)
    rp = comm_step!(ns) # returns rejection probability
    expl_step!(ns)
    return rp    
end

# run for fixed number of steps
function run!(ns::NRSTSampler{T,I,K}; nsteps::Int) where {T,I,K}
    trX  = Vector{T}(undef, nsteps)
    trIP = Vector{SVector{2,I}}(undef, nsteps) # can use a vector of SVectors since traces should not be modified
    trV  = Vector{K}(undef, nsteps)
    trRP = similar(trV)
    for n in 1:nsteps
        trX[n]  = copy(ns.x)                   # needs copy o.w. pushes a ref to ns.x
        trIP[n] = copy(ns.ip)
        trV[n]  = ns.curV[]
        trRP[n] = step!(ns)                    # note: since trIP[n] was stored before, trRP[n] is rej prob of swap **initiated** from trIP[n]
    end
    return SerialNRSTTrace(trX, trIP, trV, trRP, ns.np.N)
end

# run a tour: run the sampler until we reach the atom ip=(0,-1)
# note: by finishing at the atom (0,-1) and restarting using the renewal measure,
# repeatedly calling this function is equivalent to standard sequential sampling 
function tour!(
    ns::NRSTSampler{T,I,K};
    keep_xs::Bool = true,                 # set to false if xs can be forgotten (useful for tuning to lower mem usage) 
    ) where {T,I,K}
    renew!(ns)
    trX  = T[]
    trIP = SVector{2,I}[]                 # can use a vector of SVectors since traces should not be modified
    trV  = K[]
    trRP = K[]
    while !(ns.ip[1] == 0 && ns.ip[2] == -1)
        keep_xs && push!(trX, copy(ns.x)) # needs copy o.w. pushes a ref to ns.x
        push!(trIP, copy(ns.ip))          # same
        push!(trV, ns.curV[])
        push!(trRP, step!(ns))
    end
    keep_xs && push!(trX, copy(ns.x))
    push!(trIP, copy(ns.ip))
    push!(trV, ns.curV[])
    push!(trRP, step!(ns))
    return SerialNRSTTrace(trX, trIP, trV, trRP, ns.np.N)
end

# run multiple tours, return processed output
function run_tours!(
    ns::NRSTSampler{T,TI,TF};
    ntours::Int,
    kwargs...
    ) where {T,TI,TF}
    results = Vector{SerialNRSTTrace{T,TI,TF}}(undef, ntours)
    ProgressMeter.@showprogress 1 "Sampling: " for t in 1:ntours
        results[t] = tour!(ns;kwargs...)
    end
    return ParallelRunResults(results)
end

###############################################################################
# trace postprocessing
###############################################################################

# outer constructor that parses a trace
function SerialRunResults(tr::SerialNRSTTrace{T,I,K}) where {T,I,K}
    N      = tr.N
    xarray = [T[] for _ in 0:N] # i-th entry has samples at state i
    trVs   = [K[] for _ in 0:N] # i-th entry has Vs corresponding to xarray[i]
    visacc = zeros(I, N+1, 2)   # accumulates visits
    rpacc  = zeros(K, N+1, 2)   # accumulates rejection probs
    post_process(tr, xarray, trVs, visacc, rpacc)
    SerialRunResults(tr, xarray, trVs, visacc, rpacc)
end

function post_process(
    tr::SerialNRSTTrace{T,I,K},
    xarray::Vector{Vector{T}}, # length = N+1. i-th entry has samples at state i
    trVs::Vector{Vector{K}},   # length = N+1. i-th entry has Vs corresponding to xarray[i]
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rpacc::Matrix{K}           # size (N+1) × 2. accumulates rejection probs
    ) where {T,I,K}
    for (n, ip) in enumerate(tr.trIP)
        idx    = ip[1] + 1
        idxeps = (ip[2] == one(I) ? 1 : 2)
        visacc[idx, idxeps] += one(I)
        rpacc[idx, idxeps]  += tr.trRP[n]
        length(tr.trX) >= n && push!(xarray[idx], tr.trX[n]) # handle case keep_xs=false
        push!(trVs[idx], tr.trV[n])
    end
end


