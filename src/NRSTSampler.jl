###############################################################################
# relevant structs 
###############################################################################

# encapsulates the specifics of the inference problem
struct NRSTProblem{F,G,H,K<:AbstractFloat,A<:AbstractVector{K}}
    V::F       # energy Function
    Vref::G    # energy of reference distribution
    randref::H # produces independent sample from reference distribution
    betas::A   # vector of tempering parameters (length N)
    c::A       # vector of parameters for the pseudoprior
end

# TODO: currently this only allows for continous x. Better to
# - parametrize on the type of x, say T, which could be anything
# - also parametrize ExplorationKernels, then we force that type to match x's
struct NRSTSampler{I<:Int,K<:AbstractFloat,A<:AbstractVector{K},B<:AbstractVector{<:ExplorationKernel}}
    np::NRSTProblem        # encapsulates problem specifics
    explorers::B           # vector length N of exploration kernels
    x::A                   # current state of target variable. no need to keep it in sync all the time with explorers. they are updated at exploration step
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    nexpl::I               # number of exploration steps
end

###############################################################################
# constructors and initialization methods
###############################################################################

# tune all explorers' parameters in parallel, then adjust c
function initial_tuning!(explorers,np::NRSTProblem,nsteps::Int)
    # nsteps=5000
    @unpack c,betas,V = np
    meanV = similar(betas)
    Threads.@threads for i in eachindex(meanV)
        meanV[i] = tune!(explorers[i],V,nsteps=nsteps)
    end
    copyto!(meanV,predict(loess(betas, meanV),betas))  # use LOESS smoothing to remove kinks. note: predict is not type stable!
    trpz_apprx!(c,betas,meanV)                         # use trapezoidal approx to estimate int_0^beta db E^b[V]
end

# # test
# @code_warntype initial_tuning!(np,5000) # predict not type stable!
# initial_tuning!(np,5000)
# plot(np.betas, np.c)

# constructor that also builds an NRSTProblem, does initial tuning
function NRSTSampler(V,Vref,randref,betas,nexpl)
    x = randref()
    np = NRSTProblem(V,Vref,randref,betas,similar(betas))
    explorers = init_explorers(V,Vref,randref,betas,x)
    # tune explorations kernels and get initial c estimate 
    initial_tuning!(explorers,np,10*nexpl)
    NRSTSampler(np,explorers,x,MVector(0,1),Ref(V(x)),nexpl)
end

# constructor using a given NRSTSampler (usually already initially-tuned)
# note: "np" is fully shared, only "explorers" is deepcopied.
# so don't change np in threads!
function NRSTSampler(ns::NRSTSampler)
    x = ns.np.randref()
    explorers_copy = deepcopy(ns.explorers) # need full recursive copy, otherwise state is shared
    NRSTSampler(ns.np,explorers_copy,x,MVector(0,1),Ref(ns.np.V(x)),ns.nexpl)
end


###############################################################################
# sampling methods
###############################################################################

# communication step
function comm_step!(ns::NRSTSampler)
    @unpack np,x,ip,curV = ns
    @unpack betas,c = np
    iprop = sum(ip) # propose i+eps
    if iprop < 0 # bounce below
        ip[1]=0;ip[2]=1; return false
    elseif iprop >= length(betas) # bounce above (note base-1 indexing)
        ip[1]=length(betas)-1;ip[2]=-1; return false
    else
        i = ip[1] # current index
        # note: U(0,1) =: U < p <=> log(U) < log(p) <=> Exp(1) > -log(p) =: neglaccpr
        neglaccpr = (betas[iprop+1]-betas[i+1])*curV[] - (c[iprop+1]-c[i+1]) # Julia uses 1-based indexing
        acc = (neglaccpr < rand(Exponential())) # accept?
        if acc
            ip[1] = iprop # move
        else
            ip[2] = -ip[2] # flip direction
        end
    end
    return acc 
end

# # test
# comm_step!(ns)
# ns.ip

# exploration step
function expl_step!(ns::NRSTSampler)
    @unpack np,x,ip,curV,nexpl = ns
    @unpack V,explorers = np
    set_state!(explorers[ip[1]+1],x) # (1-based indexing)
    copyto!(x,explore!(explorers[ip[1]+1],nexpl)) # explore and update state
    curV[] = V(x) # compute energy at new point
end

# # test
# ns.x
# ns.ip
# expl_step!(ns)
# ns.x

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler)
    comm_step!(ns)
    expl_step!(ns)
end

function run!(ns::NRSTSampler, nsteps::Int)
    # nsteps=10000
    xtrace = similar(ns.x,length(ns.x),nsteps)
    iptrace = Matrix{Int}(undef,2,nsteps)
    xtrace[:,1] = ns.x
    iptrace[:,1] = ns.ip
    for n in 2:nsteps
        step!(ns)
        xtrace[:,n] = ns.x
        iptrace[:,n] = ns.ip
    end
    return xtrace,iptrace
end

# # test
# xtrace,iptrace = run!(ns,100)
# plot(iptrace[1,:],label="",seriestype = :steppost)
