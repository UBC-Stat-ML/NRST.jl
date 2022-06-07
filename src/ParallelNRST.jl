###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# method for a single NRSTSampler that creates only temp copies
# note: ns itself is never used to sample so its state should be exactly the
# same after this returns
function parallel_run(
    ns::TS;
    ntours::Int,
    kwargs...
    ) where {T,TI,TF,TS<:NRSTSampler{T,TI,TF}}
    println("\nRunning $ntours tours in parallel using $(Threads.nthreads()) threads.\n")
    res = Vector{SerialNRSTTrace{T,TI,TF}}(undef, ntours) # pre-allocate storage for results
    p   = ProgressMeter.Progress(ntours, "Sampling: ")    # prints a progress bar
    @sync for t in 1:ntours                               # @sync tells the loop to end only when all @async operations inside end
        tsk = Threads.@spawn begin
            tr = tour!(copy(ns); kwargs...)               # run a full tour with a local temp copy of ns. copying is fast relative to cost of a tour, and size(ns) is ~ size(ns.x)
            res[t] = tr                                   # writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
            ProgressMeter.next!(p)
        end
        fetch(tsk)                                        # needed so that ProgressMeter works and also so that errors are thrown: https://discourse.julialang.org/t/i-dont-get-an-error-messages-after-a-task-thread-failed/36255/2
    end
    ParallelRunResults(res)                               # post-process and return 
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
    xplapac::Vector{TF}                     # length N. accumulates explorers' acc probs
    toureff::Vector{TF}                     # tour effectiveness for each i ∈ 0:N
end
ntours(res::ParallelRunResults) = length(res.trvec)
tourlengths(res::ParallelRunResults) = nsteps.(res.trvec)

# outer constructor that parses a vector of serial traces
function ParallelRunResults(results::Vector{TST}) where {T,I,K,TST<:SerialNRSTTrace{T,I,K}}
    ntours  = length(results)
    N       = get_N(first(results))
    xarray  = [T[] for _ in 0:N]         # i-th entry has samples at level i
    trVs    = [K[] for _ in 0:N]         # i-th entry has Vs corresponding to xarray[i]
    curvis  = Matrix{I}(undef, N+1, 2)   # visits in current tour to each (i,eps) state
    sumsq   = zeros(I, N+1)              # accumulate squared number of visits for each 0:N state (for tour effectiveness)
    totvis  = zeros(I, N+1, 2)           # total visits to each (i,eps) state
    rpacc   = zeros(K, N+1, 2)           # accumulates rejection probs of swaps started from each (i,eps)
    xplapac = zeros(K, N)                # accumulates explorers' acc probs
    
    # iterate tours
    for (_, tr) in enumerate(results)
        fill!(curvis, zero(I))                                 # reset tour visits
        post_process(tr, xarray, trVs, curvis, rpacc, xplapac) # parse tour trace
        totvis .+= curvis                                      # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2                 # squared number of visits to each of 0:N (regardless of direction)
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq)    # = (sum(totvis, dims=2)/ntours).^2 ./ (sumsq/ntours)
    ParallelRunResults(results, xarray, trVs, totvis, rpacc, xplapac, toureff)    
end

