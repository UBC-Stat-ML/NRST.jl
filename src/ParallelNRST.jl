###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

struct ParallelRunResults{T,TInt<:Int,TF<:AbstractFloat} <: RunResults
    trvec::Vector{SerialNRSTTrace{T,TInt}} # vector of raw traces from each tour
    xarray::Vector{Vector{T}}              # i-th entry has samples at state i
    visits::Matrix{TInt}                   # total number of visits to each (i,eps)
    rejecs::Matrix{TInt}                   # total rejections of swaps started from each (i,eps)
    toureff::Vector{TF}                    # tour effectiveness for each i ∈ 0:N
end
ntours(res::ParallelRunResults) = length(trvec)

#######################################
# initialization methods
#######################################

# create vector of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthreads-1 (deep)copies
# note that np object is still shared since it's not modified during simulation
function copy_sampler(ns::NRSTSampler{T,I}; nthreads::I) where {T,I}
    samplers    = Vector{NRSTSampler{T,I}}(undef, nthreads)
    samplers[1] = ns
    for i in 2:nthreads
        samplers[i] = NRSTSampler(ns) # copy constructor
    end
    return samplers
end

#######################################
# sampling methods
#######################################

# run in parallel using a vector of identical copies of NRST samplers
function run!(
    samplers::Vector{NRSTSampler{T,I}}; # contains nthreads (deep)copies of an NRSTSampler object 
    ntours::I,                          # run a total of ntours tours
    verbose::Bool = false
) where {T,I}
    # need separate storage for each threadid because push! is not atomic!!!
    # each thread gets a vector of traces to push! results to
    trace_vec = [SerialNRSTTrace{T,I}[] for _ in eachindex(samplers)]

    # run tours in parallel
    Threads.@threads for tour in 1:ntours
        tid = Threads.threadid()
        tr  = tour!(samplers[tid]) # run a full tour with the sampler corresponding to this thread, avoiding race conditions
        push!(trace_vec[tid], tr)  # push results to this thread's own storage, avoiding race conditions
        verbose && println(
            "thread ", tid, ": completed tour ", tour
        )
    end
    
    trvec = vcat(trace_vec...) # collect into a single Vector{SerialNRSTTrace{T,I}}
    return post_process(trvec)
end

# method for a single NRSTSampler that creates only temp copies
function parallel_run(ns::NRSTSampler{T,I}; nthreads::I, ntours::I) where {T,I}
    run!(copy_sampler(ns, nthreads=nthreads), ntours=ntours)
end

#######################################
# trace postprocessing
#######################################

function post_process(trvec::Vector{SerialNRSTTrace{T,I}}) where {T,I}
    # allocate storage
    ntours = length(trvec)
    N      = trvec[1].N
    xarray = [T[] for _ in 0:N]       # i-th entry has samples at state i
    curvis = Matrix{I}(undef, N+1, 2) # visits in current tour to each (i,eps) state
    sumsq  = zeros(I, N+1)            # accumulate squared number of visits for each 0:N state (for tour effectiveness)
    totvis = zeros(I, N+1, 2)         # total visits to each (i,eps) state
    rejecs = zeros(I, N+1, 2)         # total rejections of swaps started in each (i,eps) state
    
    # iterate tours
    for (tour, tr) in enumerate(trvec)
        fill!(curvis, zero(I))                   # reset tour visits
        post_process(tr, xarray, curvis, rejecs) # parse tour trace
        totvis .+= curvis                        # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2   # squared number of visits to each of 0:N (regardless of eps)
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq) # == (sum(totvis, 2)/ntours).^2 ./ (sumsq/ntours)
    ParallelRunResults(trvec, xarray, totvis, rejecs, toureff)    
end

