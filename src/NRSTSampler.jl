###############################################################################
# relevant structs 
###############################################################################

# encapsulates all the specifics of the tempered problem
struct NRSTProblem{TTM<:TemperedModel,K<:AbstractFloat,A<:Vector{K},TInt<:Int,TNT<:NamedTuple}
    tm::TTM              # a TemperedModel
    N::TInt              # number of states no counting reference (N+1 in total)
    betas::A             # vector of tempering parameters (length N+1)
    c::A                 # vector of parameters for the pseudoprior
    use_mean::Bool       # should we use "mean" (true) or "median" (false) for tuning c?
    reject_big_vs::Bool  # should we use rejection sampling from the reference to avoid the region {V=inf}
    log_grid::Bool       # should we tune the grid with beta in log scale. needed when π₀{V} = inf, since here the derivative of Λ at 0 is Inf.
    nexpls::Vector{TInt} # vector of length N with number of exploration steps adequate for each level 1:N
    xplpars::Vector{TNT} # vector of length N of named tuples, holding adequate parameters to use at each level 1:N
end

# copy constructor, allows replacing tm, but keeps everything else
function NRSTProblem(oldnp::NRSTProblem, newtm)
    NRSTProblem(
        newtm,oldnp.N,oldnp.betas,oldnp.c,oldnp.use_mean,oldnp.reject_big_vs,
        oldnp.log_grid,oldnp.nexpls,oldnp.xplpars
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

###############################################################################
# constructors and initialization methods
###############################################################################

struct BadNException{TI<:Int} <: Exception
    Ncur::TI
    Nopt::TI
end
struct NonIntegrableVException <: Exception end

# constructor that also builds an NRSTProblem and does initial tuning
function NRSTSampler(
    tm::TemperedModel,
    rng::AbstractRNG;
    N::Int              = 10,
    nexpl::Int          = 10, 
    use_mean::Bool      = true,
    reject_big_vs::Bool = true,
    log_grid::Bool      = false,
    tune::Bool          = true,
    adapt_N_rounds::Int = 3, 
    kwargs...
    )
    ns      = init_sampler(tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid)
    TE, Λ   = (NaN,NaN)
    adapt_N = 0
    while tune
        try
            TE, Λ = tune!(ns, rng; check_N=(adapt_N<adapt_N_rounds), kwargs...)
            tune  = false
        catch e
            if e isa BadNException
                @warn "N=$(e.Ncur) too " * (e.Nopt > e.Ncur ? "low" : "high") * 
                      ". Setting N=$(e.Nopt) and restarting."
                N = e.Nopt
                ns = init_sampler(
                    tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid
                )
                adapt_N += 1
            elseif e isa NonIntegrableVException
                @warn "V might not be integrable under the reference. " *
                      "Adjusting the adaptation to this fact and restarting."
                log_grid = true
                ns = init_sampler(
                    tm, rng, N, nexpl, use_mean, reject_big_vs, log_grid
                )
            else
                rethrow(e)
            end
        end        
    end
    return ns, TE, Λ
end

function init_sampler(
    tm::TemperedModel,
    rng::AbstractRNG,
    N::Int,
    nexpl::Int, 
    use_mean::Bool,
    reject_big_vs::Bool,
    log_grid::Bool
    )
    betas = init_grid(N)
    x     = initx(rand(tm, rng), rng)                            # draw an initial point
    curV  = Ref(V(tm, x))
    xpl   = get_explorer(tm, x, curV)
    np    = NRSTProblem(                                         # instantiate an NRSTProblem
        tm, N, betas, similar(betas), use_mean, reject_big_vs, 
        log_grid, fill(nexpl, N), fill(params(xpl), N)
    )
    ip    = MVector(zero(N), one(N))
    return NRSTSampler(np, xpl, x, ip, curV)
end

# grid initialization
init_grid(N::Int) = collect(range(0.,1.,N+1)) # vcat(0., range(1e-8, 1., N))

# safe initialization for arrays with float entries
# robust against disruptions by heavy tailed reference distributions
function initx(pre_x::AbstractArray{TF}, rng::AbstractRNG) where {TF<:AbstractFloat}
    x = rand(rng, Uniform(-one(TF), one(TF)), size(pre_x))
    x .* (sign.(x) .* sign.(pre_x)) # quick and dirty way to respect sign constraints 
end

# constructor for a given (V,Vref,randref) triplet
function NRSTSampler(V, Vref, randref, args...;kwargs...)
    tm = SimpleTemperedModel(V, Vref, randref)
    NRSTSampler(tm,args...;kwargs...)
end

# copy-constructor, using a given NRSTSampler (usually already tuned)
function Base.copy(ns::NRSTSampler)
    newtm = copy(ns.np.tm)                   # the only element in np that we (may) need to copy
    newnp = NRSTProblem(ns.np, newtm)        # build new Problem from the old one but using new tm
    newx  = copy(ns.x)
    ncurV = Ref(ns.curV[])
    nuxpl = copy(ns.xpl, newtm, newx, ncurV) # copy ns.xpl sharing stuff with the new sampler
    NRSTSampler(newnp, nuxpl, newx, MVector(0,1), ncurV)
end

###############################################################################
# sampling methods
###############################################################################

function refreshx!(ns::NRSTSampler, rng::AbstractRNG)
    ns.curV[] = randrefmayreject!(ns.np.tm, rng, ns.x, ns.np.reject_big_vs)
end

# reset state by sampling from the renewal measure
function renew!(ns::NRSTSampler{T,I}, rng::AbstractRNG) where {T,I}
    refreshx!(ns, rng)
    ns.ip[1] = zero(I)
    ns.ip[2] = one(I)
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
function comm_step!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
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
        acc    = nlaccr < randexp(rng)          # accept? Note: U<A <=> A>U <=> -log(A) < -log(U) ~ Exp(1) 
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
function expl_step!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack np,xpl,ip = ns
    xplap = one(K)
    if ip[1] == zero(I)
        refreshx!(ns, rng)                            # sample from ref (possibly using rejection to avoid V=inf) and update curV accordingly
    else
        β      = np.betas[ip[1]+1]                    # get the β for the level
        nexpl  = np.nexpls[ip[1]]                     # get number of exploration steps needed at this level
        params = np.xplpars[ip[1]]                    # get explorer params for this level 
        xplap  = explore!(xpl, rng, β, params, nexpl) # explore for nexpl steps. note: ns.x and ns.curV are shared with xpl
    end
    return xplap
end

# NRST step = comm_step ∘ expl_step
function step!(ns::NRSTSampler, rng::AbstractRNG)
    rp    = comm_step!(ns, rng) # returns rejection probability
    xplap = expl_step!(ns, rng) # returns explorers' acceptance probability
    return rp, xplap
end

# run for fixed number of steps
function run!(ns::NRSTSampler, rng::AbstractRNG, tr::NRSTTrace)
    @unpack trX, trIP, trV, trRP, trXplAP = tr
    nsteps = length(trV)
    for n in 1:nsteps
        trX[n]    = copy(ns.x)                 # needs copy o.w. pushes a ref to ns.x
        trIP[n]   = copy(ns.ip)
        trV[n]    = ns.curV[]
        rp, xplap = step!(ns, rng)
        trRP[n]   = rp                         # note: since trIP[n] was stored before step!, trRP[n] is rej prob of swap **initiated** from trIP[n]
        l         = ns.ip[1]
        l >= 1 && push!(trXplAP[l], xplap)     # note: since comm preceeds expl, we're correctly storing the acc prob of the most recent state
    end
end
function run!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG; nsteps::Int) where {T,I,K}
    tr = NRSTTrace(T, ns.np.N, K, nsteps)
    run!(ns, rng, tr)
    return tr
end

#######################################
# touring interface
#######################################

# method that allocates a trace object
function tour!(ns::NRSTSampler{T,I,K}, rng::AbstractRNG; kwargs...) where {T,I,K}
    tr = NRSTTrace(T, ns.np.N, K)
    tour!(ns, rng, tr; kwargs...)
    return tr
end

# run a full tour, starting from a renewal, and ending at the atom
# note: by doing the above, calling this function repeatedly should give the
# same output as the sequential version.
function tour!(
    ns::NRSTSampler{T,I,K},
    rng::AbstractRNG, 
    tr::NRSTTrace; 
    kwargs...
    ) where {T,I,K}
    renew!(ns, rng)                                # init with a renewal
    while ns.ip[1] > zero(I) || ns.ip[2] == one(I)
        save_pre_step!(ns, tr; kwargs...)          # save current (x,i,ϵ,V)
        rp, xplap = step!(ns, rng)                 # do NRST step, produce (x',i',ϵ',V'), rej prob of temp step, and average xpl acc prob from expl step 
        save_post_step!(ns, tr, rp, xplap)         # save rej prob and xpl acc prob 
    end
    save_pre_step!(ns, tr; kwargs...)              # store (x,0,-1,V)
    save_post_step!(ns, tr, one(K), K(NaN))        # we know that (-1,-1) would be rejected if attempted so we store this. also, the expl step would not use an explorer; thus the NaN.
end
function save_pre_step!(ns::NRSTSampler, tr::NRSTTrace; keep_xs::Bool=true)
    @unpack trX, trIP, trV = tr
    keep_xs && push!(trX, copy(ns.x))          # needs copy o.w. pushes a ref to ns.x
    push!(trIP, copy(ns.ip))                   # same
    push!(trV, ns.curV[])
    return
end
function save_post_step!(
    ns::NRSTSampler,
    tr::NRSTTrace,
    rp::AbstractFloat, 
    xplap::AbstractFloat
    )
    push!(tr.trRP, rp)
    l = ns.ip[1]
    l >= 1 && push!(tr.trXplAP[l], xplap)
    return
end

# run multiple tours (serially), return processed output
function run_tours!(
    ns::NRSTSampler{T,TI,TF},
    rng::AbstractRNG;
    ntours::Int,
    kwargs...
    ) where {T,TI,TF}
    results = Vector{NRSTTrace{T,TI,TF}}(undef, ntours)
    ProgressMeter.@showprogress 1 "Sampling: " for t in 1:ntours
        results[t] = tour!(ns, rng;kwargs...)
    end
    return TouringRunResults(results)
end

###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# automatic determination of required number of tours
const DEFAULT_α      = .95    # probability of the mean of indicators to be inside interval
const DEFAULT_δ      = .10    # half-width of interval
const DEFAULT_TE_min = 1.5e-4 # truncate TE's below this value => max_tours ~ 1e7 for (α, δ)=(.95, .1) 
function min_ntours_TE(TE, α=DEFAULT_α, δ=DEFAULT_δ, TE_min=DEFAULT_TE_min)
    ceil(Int, (4/max(TE_min,TE)) * abs2(norminvcdf((1+α)/2) / δ))
end
const DEFAULT_MAX_TOURS = min_ntours_TE(0.)

# multithreading method
# uses a copy of ns per task, with indep state. copying is fast relative to 
# cost of a tour, and size(ns) ~ size(ns.x)
# note: ns itself is never used to sample so its state should be exactly the
# same after this returns
function parallel_run(
    ns::TS,
    rng::AbstractRNG;
    TE::AbstractFloat = NaN,
    α::AbstractFloat  = DEFAULT_α,
    δ::AbstractFloat  = DEFAULT_δ,
    ntours::Int       = -one(TI),
    keep_xs::Bool     = true,
    verbose::Bool     = true,
    check_every::Int  = 1_000,
    max_mem_use::Real = .8,
    kwargs...
    ) where {T,TI,TF,TS<:NRSTSampler{T,TI,TF}}
    GC.gc()                                                                   # setup allocates a lot so we need all mem we can get
    ntours < zero(TI) && (ntours = min_ntours_TE(TE,α,δ))
    verbose && println(
        "\nRunning $ntours tours in parallel using " *
        "$(Threads.nthreads()) threads.\n"
    )
    
    # detect and handle memory management within PBS
    jobid= get_PBS_jobid()
    ispbs= !(jobid == "")
    mlim = ispbs ? get_cgroup_mem_limit(jobid) : Inf64

    # pre-allocate traces and prngs, and then run in parallel
    res  = [NRSTTrace(T,ns.np.N,TF) for _ in 1:ntours]                        # get one empty trace for each task
    rngs = [split(rng) for _ in 1:ntours]                                     # split rng into ntours copies. must be done outside of loop because split changes rng state.
    p    = ProgressMeter.Progress(ntours; desc="Sampling: ", enabled=verbose) # prints a progress bar
    Threads.@threads for t in 1:ntours
        tour!(copy(ns), rngs[t], res[t]; keep_xs=keep_xs, kwargs...)          # run a tour with tasks' own sampler, rng, and trace, avoiding race conditions. note: writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
        ProgressMeter.next!(p)

        # if on PBS, check every 'check_every' tours if mem usage is high. If so, gc.
        if ispbs && mod(t, check_every)==0
            per_mem_used = get_cgroup_mem_usage(jobid)/mlim
            @debug "Tour $t: $(round(100*per_mem_used))% memory used."
            if per_mem_used > max_mem_use
                @debug "Calling GC.gc() due to usage above threshold"
                GC.gc()
            end
        end
    end
    TouringRunResults(res)                                                    # post-process and return 
end

# example output of the debug statements
# Sampling:  10%|████                                     |  ETA: 0:04:41┌ Debug: 76.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
# Sampling:  11%|████▌                                    |  ETA: 0:04:26┌ Debug: 92.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
# ┌ Debug: Calling GC.gc() due to usage above threshold
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:322
# Sampling:  11%|████▋                                    |  ETA: 0:07:30┌ Debug: 7.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
