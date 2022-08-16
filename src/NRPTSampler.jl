# IDEA: just use collection of N NRST samplers, since they already provide API
# for exploration plus keeping track of perm index (ip[1]), state x, and 
# corresponding V(x). Then DEO swaps amount to interchanging ip[1] 
struct NRPTSampler{TS<:NRSTSampler}
    nss::Vector{TS} # vector of N+1 NRSTSamplers
end
function NRPTSampler(ns::NRSTSampler)
    N   = ns.np.N
    nss = [copy(ns) for _ in 1:(N+1)]
    for (n, ns) in enumerate(nss)     # init samplers with identity permutation of levels 0:N
        ns.ip[1] = n-1
    end
    NRPTSampler(nss)
end
get_N(nrpt::NRPTSampler) = nrpt.nss[1].np.N
get_perm(nrpt::NRPTSampler) = [ns.ip[1] for ns in nrpt.nss]

# in-homogeneous step method
function step!(nrpt::NRPTSampler, n::Int, rng::AbstractRNG)
    deo_step!(nrpt, rng, mod(n,2))
    expl_step!(nrpt, rng)
end

# exploration step
function expl_step!(nrpt::NRPTSampler, rng::SplittableRandom)  
    N    = get_N(nrpt)
    rngs = [split(rng) for _ in 1:(N+1)] # split rng into N+1 copies. must be done outside of loop because split changes rng state.
    Threads.@threads for n in 1:(N+1)
        expl_step!(nrpt.nss[n], rngs[n])
    end
end

# deo: use i0=0 for even, i0=1 for odd 
function deo_step!(nrpt::NRPTSampler, rng::AbstractRNG, i0::Int)
    @assert Base.isbetween(0, i0, 1)
    N    = get_N(nrpt)
    per  = get_perm(nrpt) # per[i]  = level of the ith machine (per[i] ∈ 0:N). note that machines are indexed 1:(N+1)
    sper = sortperm(per)  # sper[i] = id of the machine that is in level i-1 (sper[i] ∈ 1:(N+1))
    idxs = i0:2:(N-1)     # swap-initiating indices. either {0,2,4...} or {1,3,5,...} and always strictly less than N
    for i in idxs
        # Swapping levels (i, i+1) requires swapping machines (sper[i+1], sper[i+2])
        try_swap!(nrpt.nss[sper[i+1]], nrpt.nss[sper[i+2]], rng)
    end
end

function try_swap!(ns1::NRSTSampler, ns2::NRSTSampler, rng::AbstractRNG)
    nlaccr = swap_nlaccratio(ns1, ns2) # nlaccr = -log(A), where A is accept ratio
    acc    = nlaccr < randexp(rng)     # accept? Note: U<A <=> A>U <=> -log(A) < -log(U) ~ Exp(1) 
    if acc
        i1old     = ns1.ip[1]
        ns1.ip[1] = ns2.ip[1]
        ns2.ip[1] = i1old
    end
    rp = -expm1(-max(0., nlaccr))      # rp = 1-ap = 1-exp(-max(0.,nlaccr)) = -[exp(-max(0.,nlaccr))-1] = -expm1(-max(0.,nlaccr))
    return rp
end

# compute nlaccr = -log(A), where A is accept ratio of a swap
function swap_nlaccratio(ns1::NRSTSampler, ns2::NRSTSampler)
    β₁ = ns1.xpl.curβ[]
    β₂ = ns2.xpl.curβ[]
    V₁ = ns1.curV[]
    V₂ = ns2.curV[]
    (β₁ - β₂) * (V₂ - V₁)
end