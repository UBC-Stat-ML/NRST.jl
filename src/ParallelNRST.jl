###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728
###############################################################################

function parallel_run!(
    ns::NRSTSampler{I,K,A,B},        # (must have already been initially-tuned)
    ntours::Int = 20,                # run ntours independent tours
    nthrds::Int = Threads.nthreads() # number of threads available to run tours
) where {I,K,A,B}
    # create channel containing idle nrst samplers, which can be used by workers
    # initialize with ns and additional nthrds-1 (deep)copies
    idle_nrst = Channel{Tuple{Int,NRSTSampler}}(nthrds)
    put!(idle_nrst, (1, ns))
    foreach(i -> put!(idle_nrst, (i, NRSTSampler(ns))), 2:nthrds)

    # init storage
    xtrace = A[]
    iptrace = Vector{I}[]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        i, nrst = take!(idle_nrst) # take an idle NRSTSampler. No other thread can now work on it
        xblk, ipblk = tour!(ns)
        append!(xtrace,xblk)
        append!(iptrace,ipblk)
        println("thread ", Threads.threadid(), ": ", "completed tour ", tour, " using sampler ", i)
        put!(idle_nrst, (i, nrst)) # mark resource as idle
    end

    return xtrace, iptrace
end