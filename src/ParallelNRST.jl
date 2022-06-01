###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

#######################################
# initialization methods
#######################################

# create Channel of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthreads-1 copies
# note that np object is still shared since it's not modified during simulation
function replicate(ns::NRSTSampler)
    nthreads    = Threads.nthreads()
    samplers    = Channel{typeof(ns)}(nthreads)
    put!(samplers, ns)
    for i in 2:nthreads
        put!(samplers, copy(ns)) # use custom copy constructor
    end
    return samplers
end

#######################################
# sampling methods
#######################################

# run in parallel using a vector of identical copies of NRST samplers
function run!(
    samplers::Channel{TS}; # contains nthreads (deep)copies of an NRSTSampler object 
    ntours::Int,           # total number of tours to run
    kwargs...
) where {T,TI,TF,TS<:NRSTSampler{T,TI,TF}}
    # pre-allocate storage for results
    results = Vector{SerialNRSTTrace{T,TI,TF}}(undef, ntours)

    # run tours in parallel, show progress
    p = ProgressMeter.Progress(ntours, "Sampling: ")
    @sync for t in 1:ntours           # @sync tells the loop to end only when all @async operations inside end
        Threads.@spawn begin
            ns = take!(samplers)      # take a sampler out of the idle repository
            tr = tour!(ns; kwargs...) # run a full tour with a sampler that cannot be used by other thread
            results[t] = tr           # writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
            put!(samplers, ns)        # return the sampler to the idle channel 
            ProgressMeter.next!(p)
            yield()                   # https://github.com/timholy/ProgressMeter.jl/issues/189#issuecomment-792353566
        end
    end
    
    ParallelRunResults(results)       # post-process and return 
end

# method for a single NRSTSampler that creates only temp copies
function parallel_run(ns::NRSTSampler; ntours::Int, kwargs...)
    if Threads.nthreads() == 1
        return run_tours!(ns; ntours=ntours, kwargs...)
    end
    samplers = replicate(ns)
    res = run!(samplers; ntours=ntours, kwargs...)
    close(samplers)
    return res
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
function ParallelRunResults(results::Vector{TST}) where {T,I,K,TST<:SerialNRSTTrace{T,I,K}}
    ntours = length(results)
    trvec  = Vector{TST}(undef, ntours) # storage for the raw traces
    N      = first(results).N
    xarray = [T[] for _ in 0:N]         # i-th entry has samples at level i
    trVs   = [K[] for _ in 0:N]         # i-th entry has Vs corresponding to xarray[i]
    curvis = Matrix{I}(undef, N+1, 2)   # visits in current tour to each (i,eps) state
    sumsq  = zeros(I, N+1)              # accumulate squared number of visits for each 0:N state (for tour effectiveness)
    totvis = zeros(I, N+1, 2)           # total visits to each (i,eps) state
    rpacc  = zeros(K, N+1, 2)           # accumulates rejection probs of swaps started from each (i,eps)
    
    # iterate tours
    for (tour, tr) in enumerate(results)
        fill!(curvis, zero(I))                        # reset tour visits
        post_process(tr, xarray, trVs, curvis, rpacc) # parse tour trace
        totvis .+= curvis                             # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2        # squared number of visits to each of 0:N (regardless of direction)
        trvec[tour] = tr                              # store tour trace
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq) # == (sum(totvis, 2)/ntours).^2 ./ (sumsq/ntours)
    ParallelRunResults(trvec, xarray, trVs, totvis, rpacc, toureff)    
end

