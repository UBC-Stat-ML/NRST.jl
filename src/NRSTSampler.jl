###############################################################################
# relevant structs 
###############################################################################

# encapsulates the specifics of the inference problem
struct NRSTProblem{F,G,H,K<:AbstractFloat,A<:AbstractVector{K}}
    V::F           # energy Function
    Vref::G        # energy of reference distribution
    randref::H     # produces independent sample from reference distribution
    betas::A       # vector of tempering parameters (length N)
    c::A           # vector of parameters for the pseudoprior
    use_mean::Bool # should we use "mean" (true) or "median" (false) for tuning c?
end

struct NRSTSampler{T,I<:Int,K<:AbstractFloat,B<:AbstractVector{<:ExplorationKernel},TProb}
    np::TProb              # encapsulates problem specifics
    explorers::B           # vector length N-1 of exploration kernels (N=length(betas))
    x::T                   # current state of target variable. no need to keep it in sync all the time with explorers. they are updated at exploration step
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    nexpl::I               # number of exploration steps
end

###############################################################################
# constructors and initialization methods
###############################################################################

# constructor that also builds an NRSTProblem and does initial tuning
function NRSTSampler(V, Vref, randref, betas, nexpl, use_mean)
    x         = randref()
    np        = NRSTProblem(V, Vref, randref, betas, similar(betas), use_mean)
    explorers = init_explorers(V, Vref, betas, x)
    tune!(explorers, np, nsteps=10*nexpl) # tune explorations kernels and get initial c estimate 
    NRSTSampler(np, explorers, x, MVector(0,1), Ref(V(x)), nexpl)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
# note: "np" is fully shared, only "explorers" is deepcopied.
# so don't change np in threads!
function NRSTSampler(ns::NRSTSampler)
    x = ns.np.randref()
    explorers_copy = deepcopy(ns.explorers) # need full recursive copy, otherwise state is shared
    NRSTSampler(ns.np, explorers_copy, x, MVector(0,1), Ref(ns.np.V(x)), ns.nexpl)
end

###############################################################################
# sampling methods
###############################################################################

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler)
    copyto!(ns.x,ns.np.randref())
    ns.ip[1]  = 0
    ns.ip[2]  = 1
    ns.curV[] = ns.np.V(ns.x)
end

# communication step
function comm_step!(ns::NRSTSampler)
    @unpack np,ip,curV = ns
    @unpack betas,c = np
    iprop = sum(ip)               # propose i + eps
    if iprop < 0                  # bounce below
        ip[1] = 0
        ip[2] = 1
        return false
    elseif iprop >= length(betas) # bounce above
        ip[1] = length(betas) - 1
        ip[2] = -1
        return false
    else
        i = ip[1] # current index
        # note: U(0,1) =: U < p <=> log(U) < log(p) 
        # <=> Exp(1) > -log(p) =: neglaccpr
        neglaccpr = (betas[iprop+1] - betas[i+1]) * curV[] - (c[iprop+1] - c[i+1])
        acc = (neglaccpr < rand(Exponential())) # accept?
        if acc
            ip[1] = iprop # move
        else
            ip[2] = -ip[2] # flip direction
        end
    end
    return acc 
end

# exploration step
function expl_step!(ns::NRSTSampler)
    @unpack x,np,explorers,ip,curV,nexpl = ns
    if ip[1] == 0
        copyto!(x, np.randref()) 
    else
        expl = explorers[ip[1]] # current explorer (recall that ip[1] is 0-based)
        set_state!(expl, x)     # pass current state to explorer
        explore!(expl, nexpl)   # explore for nexpl steps
        copyto!(x, expl.x)      # update nrst's state with the explorer's
    end
    curV[] = np.V(x)            # compute energy at new point
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler)
    comm_step!(ns)
    expl_step!(ns)
end

# run for fixed number of steps
function run!(ns::NRSTSampler{T,I}; nsteps::Int) where {T,I}
    xtrace  = Vector{T}(undef, nsteps)
    iptrace = Matrix{I}(undef, 2, nsteps) # can use a matrix here
    for n in 1:nsteps
        xtrace[n] = copy(ns.x)                      # needs copy o.w. pushes a ref to ns.x
        copyto!(iptrace, 1:2, n:n, ns.ip, 1:2, 1:1) # no need for another copy since there's already 1 due to implicit conversion
        step!(ns)
    end
    return xtrace, iptrace
end

# run a tour: run the sampler until we reach the atom ip=(0,-1)
# note: by finishing at the atom (0,-1) and restarting using the renewal measure,
# repeatedly calling this function is equivalent to standard sequential sampling 
function tour!(ns::NRSTSampler{T,I}) where {T,I}
    renew!(ns)
    xtrace = T[]
    iptrace = MVector{2,I}[]
    while !(ns.ip[1] == 0 && ns.ip[2] == -1)
        push!(xtrace, copy(ns.x))   # needs copy o.w. pushes a ref to ns.x
        push!(iptrace, copy(ns.ip)) # same
        step!(ns)
    end
    push!(xtrace, copy(ns.x))
    push!(iptrace, copy(ns.ip))
    return xtrace, iptrace
end
