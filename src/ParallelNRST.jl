###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# create vector of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthrds-1 (deep)copies
function copy_sampler(
    ns::NRSTSampler{T,I,K,B};        
    nthrds::Int
) where {T,I,K,B}
    samplers = Vector{NRSTSampler{T,I,K,B}}(undef, nthrds)
    samplers[1] = ns
    for i in 2:nthrds
        samplers[i] = NRSTSampler(ns) # copy constructor
    end
    return samplers
end

# run in parallel using a vector of identical copies of NRST samplers
function parallel_run!(
    samplers::Vector{NRSTSampler{T,I,K,B}}; # contains nthrds (deep)copies of an NRSTSampler object 
    ntours::Int,                            # run a total of ntours tours
    verbose::Bool = false
) where {T,I,K,B}
    # need separate storage for each threadid because push! is not atomic!
    trace_vec = [Tuple{K, Vector{T}, Vector{MVector{2,I}}}[] for _ in 1:length(samplers)]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        tstats = @timed begin
            xtrace, iptrace = tour!(samplers[Threads.threadid()]) # run a full tour with the sampler corresponding to this thread, avoiding race conditions
        end
        push!(
            trace_vec[Threads.threadid()],                        # push results to this thread's own storage, avoiding race condition
            (tstats.time - tstats.gctime, xtrace, iptrace)        # remove GC time from total elapsed time
        )
        verbose && println(
            "thread ", Threads.threadid(), ": completed tour ", tour
        )
    end
    
    trace = vcat(trace_vec...)
    N = length(samplers[1].np.betas)
    return postprocess_tours(trace, ntours, N)
end

# utility that creates the samplers vector for you and then returns it along results
function parallel_run!(
    ns::NRSTSampler;        
    ntours::Int,                      # run a total of ntours
    nthrds::Int = Threads.nthreads(), # number of threads available to run tours
    verbose::Bool = false
)
    samplers = copy_sampler(ns, nthrds = nthrds)
    results = parallel_run!(samplers, ntours=ntours, verbose=verbose)
    return (samplers = samplers, results = results)
end

# TODO: split every item into 2 separate functions
# - light postprocess: only return nsteps, times
# - full postprocess : light + the rest
# Should help some tests go faster where only light pp is needed
"""
    postprocess_tours(trace, ntours, N)

Returns a `NamedTuple` containing the fields

- `:nsteps`: vector of length `ntours` containing the total number of
             steps of each tour
- `:times` : vector of length `ntours` containing the total time spent
             in each tour
- `:xarray`: vector of length `N+1`. The `i`-th entry contains samples from the
             `i`-th annealed distribution, and its length is equal to the total
             number of visits to that level.
- `:visits`: matrix of size `ntours × N` containing the number of visits
             to each state in each tour
- `:ip`    : the full trace of the index process
"""
function postprocess_tours(
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}},
    ntours::Int,
    N::Int
    ) where {K,T,I}

    xarray = [T[] for _ in 1:N]       # sequentially collect the value of x at each of the levels
    nsteps = Vector{I}(undef, ntours) # number of steps in every tour
    times  = Vector{K}(undef, ntours) # time spent in every tour
    visits = zeros(I, ntours, N)      # number of visits to each level on each tour
    iptracefull = zeros(I, 2 ,0)      # collect the whole index process in one big matrix (for the obligatory index process plot)
    tour = 0
    for (seconds, xtrace, iptrace) in trace
        tour += 1
        len = 0
        for (n, ip) in enumerate(iptrace)
            len += 1
            visits[tour, ip[1] + 1] += 1
            push!(xarray[ip[1] + 1], xtrace[n])
        end
        nsteps[tour] = len
        times[tour]  = seconds
        iptracefull  = hcat(iptracefull, reduce(hcat, iptrace))
    end
    @assert tour == ntours
    return (
        nsteps = nsteps, times = times, xarray = xarray, visits = visits,
        ip = collect(iptracefull')
    )
end
