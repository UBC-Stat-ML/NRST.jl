###############################################################################
# Non-reversible Parallel Tempering
# Idea: just use collection of N+1 NRST samplers, since they already provide API
# for exploration plus keeping track of perm index (ip[1]), state x, and 
# corresponding V(x). Then DEO swaps amount to interchanging ip[1] 
###############################################################################

struct NRPTSampler{TS<:NRSTSampler}
    nss::Vector{TS} # vector of N+1 NRSTSamplers
end
function NRPTSampler(oldns::NRSTSampler)
    N     = oldns.np.N
    betas = oldns.np.betas
    nss   = [copy(oldns) for _ in 1:(N+1)]
    for (n, ns) in enumerate(nss)
        ns.ip[1] = n-1                                      # init sampler with identity permutation of levels 0:N
        update_β!(ns.xpl, betas[n])                         # set its explorer's beta to match (similar to replicate function)
    end
    NRPTSampler(nss)
end

# utilities
get_N(nrpt::NRPTSampler) = nrpt.nss[1].np.N
get_perm(nrpt::NRPTSampler) = [ns.ip[1] for ns in nrpt.nss]

# return explorers. for compat with current code for tuning, sorts by level and excludes beta=0
# note: since run does deo first and then expl, the params in xpls should match the ones in np.xplpars
function get_xpls(nrpt::NRPTSampler)
    per  = get_perm(nrpt)                                   # per[i]  = level of the ith machine (per[i] ∈ 0:N). note that machines are indexed 1:(N+1)
    sper = sortperm(per)                                    # sper[i] = id of the machine that is in level i-1 (sper[i] ∈ 1:(N+1))
    [nrpt.nss[i].xpl for i in sper[2:end]]                  # skip machine handling level 0
end

# struct for storing minimal info about an nsteps run
struct NRPTTrace{T, TF<:AbstractFloat, TI<:Int}
    n::Base.RefValue{TI}                                    # current step
    perms::Matrix{TI}                                       # matrix of size (N+1)×nsteps for collecting permutations
    xs::Matrix{T}                                           # matrix of size (N+1)×nsteps for collecting x samples
    Vs::Matrix{TF}                                          # matrix of size (N+1)×nsteps for collecting V values
    rpsum::Vector{TF}                                       # vector of length N, accumulates sum of rej prob of swaps started from i=0:N-1
end
function NRPTTrace(nrpt::NRPTSampler, nsteps::Int)
    N = get_N(nrpt)
    T = typeof(first(nrpt.nss).x)
    NRPTTrace(
        Ref(zero(N)), Matrix{typeof(N)}(undef, N+1, nsteps),
        Matrix{T}(undef, N+1, nsteps), 
        Matrix{Float64}(undef, N+1, nsteps), zeros(Float64, N)
    )
end
averej(tr::NRPTTrace) = tr.rpsum/(tr.n[]/2)                 # compute average rejection, using that DEO uses each swap half the time
rows2vov(M::AbstractMatrix) = [copy(r) for r in eachrow(M)] # used for converting the tr.Vs matrix to vector of vectors

# inhomogeneous NRPT step(n) = expl ∘ deo(n)
function step!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    deo_step!(nrpt, rng, tr)
    expl_step!(nrpt, rng)
    store_results!(nrpt, tr)
end

# store in trace
function store_results!(nrpt::NRPTSampler, tr::NRPTTrace)
    # copy the i-th machine's data to the l(i)+1 position in storage, where l(i)
    # is the level the i-th machine is currently in charge of 
    for (i, l) in enumerate(get_perm(nrpt))
        tr.perms[i, tr.n[]] = l
        tr.xs[l+1, tr.n[]]  = copy(nrpt.nss[i].x)
        tr.Vs[l+1, tr.n[]]  = nrpt.nss[i].curV[]
    end
end

# exploration step
# note: since NRST does not use explorer at level=0, the explorer of the machine
# at level 0 will have the pre-comm_step (stale) beta>0 and param.
function expl_step!(nrpt::NRPTSampler, rng::SplittableRandom)  
    N    = get_N(nrpt)
    rngs = [split(rng) for _ in 1:(N+1)]                    # split rng into N+1 copies. must be done outside of loop because split changes rng state.
    Threads.@threads for n in 1:(N+1)
        expl_step!(nrpt.nss[n], rngs[n])                    # this step updates the xpl beta and params only for levels>0
    end
end

# DEO. Inhomogeneous Markov step, looks at tr.n counter
function deo_step!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    N    = get_N(nrpt)
    per  = get_perm(nrpt) # per[i]  = level of the ith machine (per[i] ∈ 0:N). note that machines are indexed 1:(N+1)
    sper = sortperm(per)  # sper[i] = id of the machine that is in level i-1 (sper[i] ∈ 1:(N+1))
    i0   = mod(tr.n[], 2) # even or odd?
    idxs = i0:2:(N-1)     # lower-swap indices. either {0,2,4...} or {1,3,5,...} and always strictly less than N (there's no N+1 level to swap it with)
    for i in idxs         # this could be parallelized but the calculations are so trivial that it would be pointless
        # Swapping levels (i, i+1) requires swapping the indices of machines (sper[i+1], sper[i+2])
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
    nlaccr = -(β₂ - β₁) * (V₂ - V₁)
    # @debug "Swap ($(ns1.ip[1]),$(ns2.ip[1])), accratio=$(exp(-nlaccr)), β₁=$β₁, β₂=$β₂, V₁=$V₁, V₂=$V₂"
    return nlaccr
end

# run for fixed number of steps
function run!(nrpt::NRPTSampler, rng::AbstractRNG, tr::NRPTTrace)
    nsteps = size(tr.Vs, 2)
    for _ in 1:(nsteps-tr.n[])
        step!(nrpt, rng, tr)
    end
end
function run!(nrpt::NRPTSampler, rng::AbstractRNG, nsteps::Int)
    tr = NRPTTrace(nrpt, nsteps)
    run!(nrpt, rng, tr)
    return tr
end

# # note: to get the typical index process plot for NRPT, use
# using Plots, ColorSchemes
# nrpt = NRST.NRPTSampler(ns);
# tr = NRST.run!(nrpt,rng,100);
# N = NRST.get_N(nrpt)
# M = hcat(collect(0:N),tr.perms)
# plot(M', color_palette=palette(:thermal,N+1,rev=true), legend=false)
