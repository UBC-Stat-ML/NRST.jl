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
struct NRSTProblem{TFuns,K<:AbstractFloat,A<:AbstractVector{K},TInt<:Int}
    fns::TFuns     # struct that contains at least V, Vref, and randref
    N::TInt        # number of states additional to reference (N+1 in total)
    betas::A       # vector of tempering parameters (length N+1)
    c::A           # vector of parameters for the pseudoprior
    use_mean::Bool # should we use "mean" (true) or "median" (false) for tuning c?
end

# copy constructor, allows replacing fns, but keeps everything else
function NRSTProblem(oldnp::NRSTProblem, newfns)
    NRSTProblem(newfns,oldnp.N,oldnp.betas,oldnp.c,oldnp.use_mean)
end

# struct for the sampler
struct NRSTSampler{T,I<:Int,K<:AbstractFloat,B<:AbstractVector{<:ExplorationKernel},TProb}
    np::TProb              # encapsulates problem specifics
    explorers::B           # vector length N of exploration kernels
    x::T                   # current state of target variable. no need to keep it in sync all the time with explorers. they are updated at exploration step
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    nexpl::I               # number of exploration steps
end

# raw trace of a serial run
struct SerialNRSTTrace{T,TInt<:Int}
    xtrace::Vector{T}
    iptrace::Vector{MVector{2,TInt}}
    acctrace::BitVector
    N::TInt
end
nsteps(tr::SerialNRSTTrace) = length(tr.xtrace) # recover nsteps

# processed output
struct SerialRunResults{T,TInt<:Int} <: RunResults
    tr::SerialNRSTTrace{T,TInt} # raw trace
    xarray::Vector{Vector{T}}   # i-th entry has samples at state i
    visits::Matrix{TInt}        # total number of visits to each (i,eps)
    rejecs::Matrix{TInt}        # total rejections of swaps started from each (i,eps)
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
    init::Bool     = true,
    verbose::Bool  = false
    )
    if ismissing(betas)
        betas = pushfirst!(2. .^ range(-min(23, N-1), 0, N), 0.)  # initialize with exponentially spaced grid
    else
        N = length(betas) - 1
    end
    np    = NRSTProblem(fns, N, betas, similar(betas), use_mean)  # instantiate an NRSTProblem
    x     = fns.randref()                                         # draw an initial point
    es    = init_explorers(fns, betas, x)                         # instantiate a vector of N explorers
    ns    = NRSTSampler(np, es, x, MVector(0,1), Ref(fns.V(x)), nexpl)
    init && initialize!(ns, nsteps = 10*nexpl, verbose = verbose) # tune explorers and get initial c estimate
    return ns
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    fns = SimpleFuns(V, Vref, randref)
    NRSTSampler(fns,args...;kwargs...)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
function NRSTSampler(ns::NRSTSampler)
    newfns    = copy(ns.np.fns)            # the only element in np that we copy
    newnp     = NRSTProblem(ns.np, newfns) # build new Problem using new fns
    newx      = newfns.randref()
    explorers = init_explorers(newfns, newnp.betas, newx)
    NRSTSampler(
        newnp, explorers, newx, MVector(0,1), Ref(newfns.V(newx)), ns.nexpl
    )
end

###############################################################################
# sampling methods
###############################################################################

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler)
    copyto!(ns.x,ns.np.fns.randref())
    ns.ip[1]  = 0
    ns.ip[2]  = 1
    ns.curV[] = ns.np.fns.V(ns.x)
end

# communication step
function comm_step!(ns::NRSTSampler)
    @unpack np,ip,curV = ns
    @unpack N,betas,c = np
    iprop = sum(ip)               # propose i + eps
    if iprop < 0                  # bounce below
        ip[1] = 0
        ip[2] = 1
        return false
    elseif iprop > N              # bounce above
        ip[1] = N
        ip[2] = -1
        return false
    else
        i = ip[1]                 # current index
        # note: U(0,1) =: U < p <=> log(U) < log(p) <=> Exp(1) > -log(p) =: neglaccpr
        neglaccpr = (betas[iprop+1] - betas[i+1]) * curV[] - (c[iprop+1] - c[i+1])
        acc = (neglaccpr < rand(Exponential())) # accept?
        if acc
            ip[1] = iprop         # move
        else
            ip[2] = -ip[2]        # flip direction
        end
    end
    return acc 
end

# exploration step
function expl_step!(ns::NRSTSampler)
    @unpack x,np,explorers,ip,curV,nexpl = ns
    if ip[1] == 0
        copyto!(x, np.fns.randref()) 
    else
        expl = explorers[ip[1]] # current explorer (recall that ip[1] is 0-based)
        set_state!(expl, x)     # pass current state to explorer
        explore!(expl, nexpl)   # explore for nexpl steps
        copyto!(x, expl.x)      # update ns' state with the explorer's
    end
    curV[] = np.fns.V(x)        # compute energy at new point
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler)
    acc = comm_step!(ns) # returns acceptance indicator
    expl_step!(ns)
    return acc    
end

# run for fixed number of steps
function run!(ns::NRSTSampler{T,I}; nsteps::Int) where {T,I}
    xtrace   = Vector{T}(undef, nsteps)
    iptrace  = Vector{typeof(ns.ip)}(undef, nsteps)
    acctrace = BitVector(undef, nsteps)
    for n in 1:nsteps
        xtrace[n]   = copy(ns.x)  # needs copy o.w. pushes a ref to ns.x
        iptrace[n]  = copy(ns.ip)
        # copyto!(iptrace, 1:2, n:n, ns.ip, 1:2, 1:1) # no need for another copy since there's already 1 due to implicit conversion
        acctrace[n] = step!(ns)   # note: since iptrace[n] was stored before, acctrace[n] is acc of swap **initiated** from iptrace[n]
    end
    return SerialNRSTTrace(xtrace, iptrace, acctrace, ns.np.N)
end

# run a tour: run the sampler until we reach the atom ip=(0,-1)
# note: by finishing at the atom (0,-1) and restarting using the renewal measure,
# repeatedly calling this function is equivalent to standard sequential sampling 
function tour!(ns::NRSTSampler{T,I}) where {T,I}
    renew!(ns)
    xtrace   = T[]
    iptrace  = typeof(ns.ip)[]
    acctrace = BitVector()
    while !(ns.ip[1] == 0 && ns.ip[2] == -1)
        push!(xtrace, copy(ns.x))   # needs copy o.w. pushes a ref to ns.x
        push!(iptrace, copy(ns.ip)) # same
        push!(acctrace, step!(ns))
    end
    push!(xtrace, copy(ns.x))
    push!(iptrace, copy(ns.ip))
    push!(acctrace, step!(ns))
    return SerialNRSTTrace(xtrace, iptrace, acctrace, ns.np.N)
end

###############################################################################
# trace postprocessing
###############################################################################

function post_process(
    tr::SerialNRSTTrace{T,I},
    xarray::Vector{Vector{T}}, # length = N+1. i-th entry has samples at state i
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rejacc::Matrix{I}          # size (N+1) × 2. accumulates rejections
    ) where {T,I}
    for (n, ip) in enumerate(tr.iptrace)
        idx    = ip[1] + 1
        idxeps = (ip[2] == one(I) ? 1 : 2)
        visacc[idx, idxeps] += 1
        (!tr.acctrace[n]) && (rejacc[idx, idxeps] += 1)
        push!(xarray[idx], tr.xtrace[n])
    end
end
function post_process(tr::SerialNRSTTrace{T,I}) where {T,I}
    N      = tr.N
    xarray = [T[] for _ in 0:N] # i-th entry has samples at state i
    visacc = zeros(I, N+1, 2)   # accumulates visits
    rejacc = zeros(I, N+1, 2)   # accumulates rejections
    post_process(tr,xarray,visacc,rejacc)
    SerialRunResults(tr, xarray, visacc, rejacc)
end

