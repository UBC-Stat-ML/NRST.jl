###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

#######################################
# initialization methods
#######################################

# create vector of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthreads-1 (deep)copies
# note that np object is still shared since it's not modified during simulation
function copy_sampler(ns::NRSTSampler; nthreads::Int = Threads.nthreads())
    samplers    = Vector{typeof(ns)}(undef, nthreads)
    samplers[1] = ns
    for i in 2:nthreads
        samplers[i] = copy(ns) # copy constructor
    end
    return samplers
end

#######################################
# sampling methods
#######################################

# run in parallel using a vector of identical copies of NRST samplers
function run!(
    samplers::Vector{TS}; # contains nthreads (deep)copies of an NRSTSampler object 
    ntours::Int,          # total number of tours to run
    verbose::Bool = false
) where {T,TI,TF,TS<:NRSTSampler{T,TI,TF}}
    # need separate storage for each threadid because push! is not atomic!!!
    # each thread gets a vector of traces to push! results to
    trace_vec = [SerialNRSTTrace{T,TI,TF}[] for _ in samplers]

    # run tours in parallel, show progress
    p = ProgressMeter.Progress(ntours, "Sampling: ")
    Threads.@threads for tour in 1:ntours
        tid = Threads.threadid()
        tr  = tour!(samplers[tid])      # run a full tour with the sampler corresponding to this thread, avoiding race conditions
        push!(trace_vec[tid], tr)       # push results to this thread's own storage, avoiding race conditions
        verbose && println(
            "thread ", tid, ": completed tour ", tour
        )
        ProgressMeter.next!(p)
    end
    
    trvec = vcat(trace_vec...) # collect into a single Vector{SerialNRSTTrace{T,I}}
    return ParallelRunResults(trvec)
end

# method for a single NRSTSampler that creates only temp copies
function parallel_run(
    ns::NRSTSampler; 
    ntours::Int,
    nthreads::Int = Threads.nthreads()
    )
    run!(copy_sampler(ns, nthreads=nthreads), ntours=ntours)
end

#######################################
# trace postprocessing
#######################################

struct ParallelRunResults{T,TI<:Int,TF<:AbstractFloat} <: RunResults
    trvec::Vector{SerialNRSTTrace{T,TI,TF}} # vector of raw traces from each tour
    xarray::Vector{Vector{T}}               # length = N+1. i-th entry has samples at level i
    trVs::Vector{Vector{TF}}                # length = N+1. i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}                      # total number of visits to each (i,eps)
    rpacc::Matrix{TF}                       # accumulates rejection probs of swaps started from each (i,eps)
    toureff::Vector{TF}                     # tour effectiveness for each i ∈ 0:N
end
ntours(res::ParallelRunResults) = length(res.trvec)
tourlengths(res::ParallelRunResults) = nsteps.(res.trvec)

# outer constructor that parses a vector of serial traces
function ParallelRunResults(trvec::Vector{SerialNRSTTrace{T,I,K}}) where {T,I,K}
    # allocate storage
    ntours = length(trvec)
    N      = trvec[1].N
    xarray = [T[] for _ in 0:N]       # i-th entry has samples at level i
    trVs   = [K[] for _ in 0:N]       # i-th entry has Vs corresponding to xarray[i]
    curvis = Matrix{I}(undef, N+1, 2) # visits in current tour to each (i,eps) state
    sumsq  = zeros(I, N+1)            # accumulate squared number of visits for each 0:N state (for tour effectiveness)
    totvis = zeros(I, N+1, 2)         # total visits to each (i,eps) state
    rpacc  = zeros(K, N+1, 2)         # accumulates rejection probs of swaps started from each (i,eps)
    # iterate tours
    for tr in trvec
        fill!(curvis, zero(I))                        # reset tour visits
        post_process(tr, xarray, trVs, curvis, rpacc) # parse tour trace
        totvis .+= curvis                             # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2        # squared number of visits to each of 0:N (regardless of direction)
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq) # == (sum(totvis, 2)/ntours).^2 ./ (sumsq/ntours)
    ParallelRunResults(trvec, xarray, trVs, totvis, rpacc, toureff)    
end

