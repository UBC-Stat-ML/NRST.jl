###############################################################################
# relevant structs 
###############################################################################

# storage for functions associated to a tempered problem
struct SimpleFuns{TV,TVr,Tr} <: Funs
    V::TV       # energy Function
    Vref::TVr   # energy of reference distribution
    randref::Tr # produces independent sample from reference distribution
end

# default function to generate a tempered potential closure from a Funs object
# Note: Vβ captures the whole "betas" vector, so we can tune it outside and it
# will affect the outcome of Vβs generated prior to the change   
function gen_Vβ(fns::Funs, ind::Int, betas::AbstractVector{<:AbstractFloat})
    function Vβ(x)
        β = betas[ind]
        fns.Vref(x) + β*fns.V(x)
    end
end
Base.copy(fns::SimpleFuns) = fns # dont do anything

# encapsulates all the specifics of the tempered problem
# NOTE: when parallelizing, all fields must be shared except possibly
# for "fns".
struct NRSTProblem{TFuns,K<:AbstractFloat,A<:Vector{K},TInt<:Int}
    fns::TFuns           # struct that contains at least V, Vref, and randref
    N::TInt              # number of states additional to reference (N+1 in total)
    betas::A             # vector of tempering parameters (length N+1)
    c::A                 # vector of parameters for the pseudoprior
    use_mean::Bool       # should we use "mean" (true) or "median" (false) for tuning c?
    nexpls::Vector{TInt} # number of exploration steps for each explorer
end

# copy constructor, allows replacing fns, but keeps everything else
function NRSTProblem(oldnp::NRSTProblem, newfns)
    NRSTProblem(newfns,oldnp.N,oldnp.betas,oldnp.c,oldnp.use_mean,oldnp.nexpls)
end

# struct for the sampler
struct NRSTSampler{T,I<:Int,K<:AbstractFloat,B<:AbstractVector{<:ExplorationKernel},TProb}
    np::TProb              # encapsulates problem specifics
    explorers::B           # vector length N of exploration kernels
    x::T                   # current state of target variable. no need to keep it in sync all the time with explorers. they are updated at exploration step
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

# raw trace of a serial run
struct SerialNRSTTrace{T,TI<:Int,TF<:AbstractFloat}
    xtrace::Vector{T}
    iptrace::Vector{MVector{2,TI}}
    aptrace::Vector{TF}
    N::TI
end
nsteps(tr::SerialNRSTTrace) = length(tr.xtrace) # recover nsteps

# processed output
struct SerialRunResults{T,TI<:Int,TF<:AbstractFloat} <: RunResults
    tr::SerialNRSTTrace{T,TI} # raw trace
    xarray::Vector{Vector{T}} # i-th entry has samples at state i
    visits::Matrix{TI}        # total number of visits to each (i,eps)
    rpacc::Matrix{TF}         # accumulates rejection probs of swaps started from each (i,eps)
end

###############################################################################
# constructors and initialization methods
###############################################################################

# constructor that also builds an NRSTProblem and does initial tuning
function NRSTSampler(
    fns::Funs;
    betas          = missing,
    N::Int         = 15,
    nexpl::Int     = 50,
    use_mean::Bool = true,
    tune::Bool     = true,
    verbose::Bool  = false,
    kwargs...
    )
    if ismissing(betas)
        betas = init_grid(N)
    else
        N = length(betas) - 1
    end
    np = NRSTProblem(fns, N, betas, similar(betas), use_mean, fill(nexpl,N)) # instantiate an NRSTProblem
    x  = initx(fns.randref())                                                # draw an initial point
    es = init_explorers(fns, betas, x)                                       # instantiate a vector of N explorers
    ns = NRSTSampler(np, es, x, MVector(0,1), Ref(fns.V(x)))
    tune && tune!(ns; verbose = verbose, kwargs...)                          # tune explorers, c, and betas
    return ns
end

init_grid(N::Int) = collect(range(0,1,N+1))

# safe initialization for arrays with entries in reals
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TR}) where {TR<:Real}
    uno = one(TR)
    rand(Uniform(-uno,uno), size(pre_x))
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    fns = SimpleFuns(V, Vref, randref)
    NRSTSampler(fns,args...;kwargs...)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
function NRSTSampler(ns::NRSTSampler)
    newfns = copy(ns.np.fns)            # the only element in np that we copy
    newnp  = NRSTProblem(ns.np, newfns) # build new Problem using new fns
    newx   = newfns.randref()
    expls  = init_explorers(newfns, newnp.betas, newx)
    NRSTSampler(newnp, expls, newx, MVector(0,1), Ref(newfns.V(newx)))
end

###############################################################################
# sampling methods
###############################################################################

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler{T,I,K}) where {T,I,K}
    copyto!(ns.x,ns.np.fns.randref())
    ns.ip[1]  = zero(I)
    ns.ip[2]  = one(I)
    ns.curV[] = ns.np.fns.V(ns.x)
end

# communication step
function comm_step!(ns::NRSTSampler{T,I,K}) where {T,I,K}
    @unpack np,ip,curV = ns
    @unpack N,betas,c = np
    iprop = sum(ip)               # propose i + eps
    if iprop < 0                  # bounce below
        ip[1] = zero(I)
        ip[2] = one(I)
        return zero(K)
    elseif iprop > N              # bounce above
        ip[1] = N
        ip[2] = -one(I)
        return zero(K)
    else
        i = ip[1]                 # current index
        # note: U(0,1) =: U < R <=> log(U) < log(R) <=> Exp(1) > -log(R) =: nlaccr
        nlaccr = (betas[iprop+1] - betas[i+1])*curV[] - (c[iprop+1] - c[i+1])
        acc    = (nlaccr < rand(Exponential())) # accept?
        if acc
            ip[1] = iprop         # move
        else
            ip[2] = -ip[2]        # flip direction
        end
    end
    exp(-max(zero(K), nlaccr))    # ap = min(1.,exp(-nlaccr)) = exp(min(0.,-nlaccr)) = exp(-max(0.,nlaccr))
end

# exploration step
function expl_step!(ns::NRSTSampler{T,I,K}) where {T,I,K}
    @unpack x,np,explorers,ip,curV = ns
    if ip[1] == zero(I)
        copyto!(x, np.fns.randref()) 
    else
        expl  = explorers[ip[1]] # current explorer (recall that ip[1] is 0-based)
        nexpl = np.nexpls[ip[1]] # get the exporer's number of exploration steps
        set_state!(expl, x)      # pass current state to explorer
        explore!(expl, nexpl)    # explore for nexpl steps
        copyto!(x, expl.x)       # update ns' state with the explorer's
    end
    curV[] = np.fns.V(x)         # compute energy at new point
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler)
    ap = comm_step!(ns) # returns acceptance probability
    expl_step!(ns)
    return ap    
end

# run for fixed number of steps
function run!(ns::NRSTSampler{T,I,K}; nsteps::Int) where {T,I,K}
    xtrace  = Vector{T}(undef, nsteps)
    iptrace = Vector{typeof(ns.ip)}(undef, nsteps)
    aptrace = Vector{K}(undef, nsteps)
    for n in 1:nsteps
        xtrace[n]  = copy(ns.x)  # needs copy o.w. pushes a ref to ns.x
        iptrace[n] = copy(ns.ip)
        aptrace[n] = step!(ns)   # note: since iptrace[n] was stored before, aptrace[n] is acc prob of swap **initiated** from iptrace[n]
    end
    return SerialNRSTTrace(xtrace, iptrace, aptrace, ns.np.N)
end

# run a tour: run the sampler until we reach the atom ip=(0,-1)
# note: by finishing at the atom (0,-1) and restarting using the renewal measure,
# repeatedly calling this function is equivalent to standard sequential sampling 
function tour!(ns::NRSTSampler{T,I,K}) where {T,I,K}
    renew!(ns)
    xtrace  = T[]
    iptrace = typeof(ns.ip)[]
    aptrace = K[]
    while !(ns.ip[1] == 0 && ns.ip[2] == -1)
        push!(xtrace, copy(ns.x))   # needs copy o.w. pushes a ref to ns.x
        push!(iptrace, copy(ns.ip)) # same
        push!(aptrace, step!(ns))
    end
    push!(xtrace, copy(ns.x))
    push!(iptrace, copy(ns.ip))
    push!(aptrace, step!(ns))
    return SerialNRSTTrace(xtrace, iptrace, aptrace, ns.np.N)
end

###############################################################################
# trace postprocessing
###############################################################################

function post_process(
    tr::SerialNRSTTrace{T,I,K},
    xarray::Vector{Vector{T}}, # length = N+1. i-th entry has samples at state i
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rpacc::Matrix{K}           # size (N+1) × 2. accumulates rejection probs
    ) where {T,I,K}
    for (n, ip) in enumerate(tr.iptrace)
        idx    = ip[1] + 1
        idxeps = (ip[2] == one(I) ? 1 : 2)
        visacc[idx, idxeps] += one(I)
        rpacc[idx, idxeps]  += one(K) - tr.aptrace[n]
        push!(xarray[idx], tr.xtrace[n])
    end
end
function post_process(tr::SerialNRSTTrace{T,I,K}) where {T,I,K}
    N      = tr.N
    xarray = [T[] for _ in 0:N] # i-th entry has samples at state i
    visacc = zeros(I, N+1, 2)   # accumulates visits
    rpacc  = zeros(K, N+1, 2)   # accumulates rejection probs
    post_process(tr,xarray,visacc,rpacc)
    SerialRunResults(tr, xarray, visacc, rpacc)
end

