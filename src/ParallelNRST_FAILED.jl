###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728
# FAILED APPROACH: channels are not thread safe? Consider filing an issue
###############################################################################

function parallel_run!(
    ns::NRSTSampler{T,I,K,B};        
    ntours::Int,                      # run a total of ntours
    nthrds::Int = Threads.nthreads(), # number of threads available to run tours
    verbose::Bool = false
) where {T,I,K,B}
    # create channel containing idle nrst samplers, which can be used by workers
    # initialize with ns and additional nthrds-1 (deep)copies
    idle_nrst = Channel{Tuple{Int,NRSTSampler{T,I,K,B}}}(nthrds)
    put!(idle_nrst, (1, ns))
    for i in 2:nthrds
        put!(idle_nrst, (i, NRSTSampler(ns)))
    end

    # returns channel and output
    return (chan = idle_nrst, trace = parallel_run!(idle_nrst, ntours=ntours, verbose=verbose))
end

function parallel_run!(
    idle_nrst::Channel{Tuple{Int,NRSTSampler{T,I,K,B}}}; # contains (deep)copies of an NRSTSampler object 
    ntours::Int,                                         # run a total of ntours tours
    verbose::Bool = false
) where {T,I,K,B}
    # need separate storage for each threadid because push! is not atomic!
    trace_vec = [Tuple{K, Vector{T}, Vector{MVector{2,I}}}[] for _ in 1:idle_nrst.sz_max]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        i, nrst = take!(idle_nrst)         # take an idle NRSTSampler. No other thread can now work on it
        seconds = @elapsed begin
            xtrace, iptrace = tour!(nrst)  # run a full tour with nrst
        end
        push!(
            trace_vec[Threads.threadid()], # push results to this thread's own storage, avoiding race condition
            (seconds, xtrace, iptrace)
        )
        verbose && println(
            "thread ", Threads.threadid(), ": completed tour ", tour, " using sampler ", i
        )
        put!(idle_nrst, (i, nrst))         # mark nrst as idle
    end
    
    trace = vcat(trace_vec...) # collect traces into unique vector
    i, nrst = take!(idle_nrst) # cumbersome way of finding what N is...
    N = length(nrst.np.betas)
    put!(idle_nrst, (i, nrst))
    return postprocess_tours(ntours, N, trace)
end

# process the raw trace
function postprocess_tours(
    ntours::Int,
    N::Int,
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}}
    ) where {K,T,I}

    xarray = [T[] for _ in 1:N]       # sequentially collect the value of x at each of the levels
    nsteps = Vector{I}(undef, ntours) # number of steps in every tour
    times  = Vector{K}(undef, ntours) # time spent in every tour
    visits = zeros(I, ntours, N)      # number of visits to each level on each tour
    iptracefull = zeros(I, 2 ,0)      # collect the whole index process in one big matrix (for the obligatory plot)
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
