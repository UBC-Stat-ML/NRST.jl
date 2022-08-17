# IDEA: just use collection of N+1 NRST samplers, since they already provide API
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

# struct for storing minimal info about an nsteps run
struct NRPTTrace{TF<:AbstractFloat,TI<:Int}
    n::Base.RefValue{TI} # current step
    Vs::Matrix{TF}       # matrix of size (N+1)×nsteps for collecting V values 
    rpsum::Vector{TF}    # vector of length N, accumulates sum of rej prob of swaps started from i=0:N-1
end
function NRPTTrace(N::Int, nsteps::Int)
    NRPTTrace(Ref(0), Matrix{Float64}(undef, N+1, nsteps), zeros(Float64, N))
end
Base.size(tr::NRPTTrace) = size(tr.Vs)

# inhomogeneous NRPT step(n) = expl ∘ deo(n)
function step!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    deo_step!(nrpt, rng, tr)
    expl_step!(nrpt, rng)
    store_results!(nrpt, tr)
end

# store in trace
function store_results!(nrpt::NRPTSampler, tr::NRPTTrace)
    for (i, ns) in enumerate(nrpt.nss)
        tr.Vs[i, tr.n[]] = ns.curV[]
    end
end

# exploration step
function expl_step!(nrpt::NRPTSampler, rng::SplittableRandom)  
    N    = get_N(nrpt)
    rngs = [split(rng) for _ in 1:(N+1)] # split rng into N+1 copies. must be done outside of loop because split changes rng state.
    Threads.@threads for n in 1:(N+1)
        expl_step!(nrpt.nss[n], rngs[n])
    end
end

# DEO. Inhomogeneous Markov step, looks at tr.n counter
function deo_step!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    N    = get_N(nrpt)
    per  = get_perm(nrpt) # per[i]  = level of the ith machine (per[i] ∈ 0:N). note that machines are indexed 1:(N+1)
    sper = sortperm(per)  # sper[i] = id of the machine that is in level i-1 (sper[i] ∈ 1:(N+1))
    i0   = mod(tr.n[], 2) # even or odd?
    idxs = i0:2:(N-1)     # lower-swap indices. either {0,2,4...} or {1,3,5,...} and always strictly less than N (there's no N+1 level)
    for i in idxs         # this could be parallelized but the calculations are so trivial that it would be pointless
        # Swapping levels (i, i+1) requires swapping machines (sper[i+1], sper[i+2])
        tr.rpsum[i+1] += try_swap!(nrpt.nss[sper[i+1]], nrpt.nss[sper[i+2]], rng)
    end
    tr.n[] += 1           # increase step counter in trace
end

# attempt swapping indices between (ns1,ns2)
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
    β₁ = ns1.np.betas[ns1.ip[1]+1]
    β₂ = ns2.np.betas[ns2.ip[1]+1]
    V₁ = ns1.curV[]
    V₂ = ns2.curV[]
    nlaccr = (β₁ - β₂) * (V₂ - V₁)
    # @debug "Swap ($(ns1.ip[1]),$(ns2.ip[1])), accratio=$(exp(-nlaccr)), β₁=$β₁, β₂=$β₂, V₁=$V₁, V₂=$V₂"
    return nlaccr
end

# run for fixed number of steps
function run!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    _, nsteps = size(tr)
    for _ in 1:(nsteps-tr.n[])
        step!(nrpt, rng, tr)
    end
end
function run!(nrpt::NRPTSampler, rng::AbstractRNG, nsteps::Int)
    tr = NRPTTrace(get_N(nrpt), nsteps)
    run!(nrpt, rng, tr)
    return tr
end