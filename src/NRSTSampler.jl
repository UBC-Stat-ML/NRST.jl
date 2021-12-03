###############################################################################
# TODO:
# - implement second stage tuning, and keep track of mean and median of V
###############################################################################



###############################################################################
# struct defining an NRST sampler 
###############################################################################

struct NRSTSampler{I<:Int,K<:AbstractFloat,A<:AbstractVector{K}}
    np::NRSTProblem # encapsulates problem specifics
    x::A # current state of target variable
    ip::MVector{2,I} # current state of the Index Process (i,eps). uses statically sized (but mutable) vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    nexpl::I # number of exploration steps
end

# simple outer constructor
function NRSTSampler(V,Vref,randref,betas,nexpl)
    x=randref()
    np=NRSTProblem(V,Vref,randref,betas,x) # tune expl kernels
    initial_tuning!(np,10*nexpl) # tune explorations kernels and get initial c estimate
    NRSTSampler(np,x,MVector(0,1),Ref(V(x)),nexpl)
end

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
    @unpack explorers = np
    set_state!(explorers[ip[1]+1],x) # (1-based indexing)
    copyto!(x,explore!(explorers[ip[1]+1],nexpl)) # explore and update state
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
