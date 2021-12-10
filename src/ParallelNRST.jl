###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728
###############################################################################

function parallel_run!(
    ns::NRSTSampler{T,I,K,B};        
    ntours::Int,                     # run ntours independent tours
    nthrds::Int = Threads.nthreads() # number of threads available to run tours
) where {T,I,K,B}
    # create channel containing idle nrst samplers, which can be used by workers
    # initialize with ns and additional nthrds-1 (deep)copies
    idle_nrst = Channel{Tuple{Int,NRSTSampler}}(nthrds)
    put!(idle_nrst, (1, ns))
    for i in 2:nthrds
        put!(idle_nrst, (i, NRSTSampler(ns)))
    end

    # init storage: vector of tuples, each one containing the output of a single call to tour!
    trace = Tuple{Vector{T}, Vector{MVector{2,I}}}[]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        i, nrst = take!(idle_nrst) # take an idle NRSTSampler. No other thread can now work on it
        xtrace, iptrace = tour!(nrst)  # run a full tour with nrst
        push!(trace, (xtrace, iptrace))
        println("thread ", Threads.threadid(), ": completed tour ", tour, " using sampler ", i)
        put!(idle_nrst, (i, nrst)) # mark nrst as idle
    end

    return analyze_tours(length(ns.np.betas),trace)
end

# compute tour_stats: for each state i: E[#visits(i)], E[#visits(i)^2], TE[i] = E[#visits(i)]^2/E[#visits(i)^2]
# output: (tour_stats, merged x trace, merged ip trace)
function analyze_tours(
    N::Int,
    trace::Vector{Tuple{Vector{T}, Vector{MVector{2,I}}}}
    ) where {T,I}
    vs_sum = zeros(I,N)          # stores sum of vs and visits^2
    vs_sq_sum = zeros(I,N)       # stores sum of vs^2
    vs = Vector{I}(undef,N)      # number of visits for current tour
    xtracefull = T[]             
    iptracefull = zeros(I,2,0)   # can use a matrix here
    ntours = 0
    for (xtrace, iptrace) in trace
        ntours += 1

        # compute number of visits to states in this tour and update accumulators
        fill!(vs, zero(vs[1]))
        for ip in iptrace
            vs[ip[1] + 1] += 1 # 1-based indexing
        end
        vs_sum    .+= vs
        vs_sq_sum .+= (vs .* vs)

        # append traces
        append!(xtracefull, xtrace)
        iptracefull = hcat(iptracefull, reduce(hcat,iptrace))
    end
    # compute means from accumulated sums
    vs_mean    = vs_sum    ./ ntours
    vs_sq_mean = vs_sq_sum ./ ntours
    tour_stats = [
        vs_mean vs_sq_mean ((vs_mean .* vs_mean) ./ vs_sq_mean)
    ]
    return (tour_stats = tour_stats, x = xtracefull, ip = iptracefull)
end
