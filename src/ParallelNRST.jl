###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728
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
    return postprocess_tours(N, trace)
end

# compute tour_stats: for each state i: E[#visits(i)], E[#visits(i)^2], TE[i] = E[#visits(i)]^2/E[#visits(i)^2]
# output: (tour_stats, seconds per tour, merged x trace, merged ip trace)
function postprocess_tours(
    N::Int,
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}}
    ) where {K,T,I}

    vs_sum = zeros(I,N)        # stores sum of vs and visits^2
    vs_sq_sum = zeros(I,N)     # stores sum of vs^2
    vs = Vector{I}(undef,N)    # number of visits for current tour
    timetrace = K[]
    xtracefull = T[]             
    iptracefull = zeros(I,2,0) # can use a matrix here
    ntours = 0

    for (seconds, xtrace, iptrace) in trace
        ntours += 1
        
        # compute number of visits to states in this tour and update accumulators
        fill!(vs, zero(I))
        for ip in iptrace
            vs[ip[1] + 1] += 1 # 1-based indexing
        end
        vs_sum    .+= vs
        vs_sq_sum .+= (vs .* vs)

        # append to traces
        push!(timetrace, seconds)
        append!(xtracefull, xtrace)
        iptracefull = hcat(iptracefull, reduce(hcat, iptrace))
    end

    # compute means from accumulated sums
    vs_mean    = vs_sum    ./ ntours
    vs_sq_mean = vs_sq_sum ./ ntours
    tour_stats = [
        vs_mean vs_sq_mean ((vs_mean .* vs_mean) ./ vs_sq_mean)
    ]

    return (tour_stats = tour_stats, time = timetrace, x = xtracefull, ip = collect(iptracefull'))
end
