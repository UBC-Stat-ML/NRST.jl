###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728
###############################################################################

function parallel_run!(
    ns::NRSTSampler{I,K,A,B};        # (must have already been initially-tuned)
    ntours::Int,                     # run ntours independent tours
    nthrds::Int = Threads.nthreads() # number of threads available to run tours
) where {I,K,A,B}
    # create channel containing idle nrst samplers, which can be used by workers
    # initialize with ns and additional nthrds-1 (deep)copies
    idle_nrst = Channel{Tuple{Int,NRSTSampler}}(nthrds)
    put!(idle_nrst, (1, ns))
    foreach(i -> put!(idle_nrst, (i, NRSTSampler(ns))), 2:nthrds)

    # init storage
    # TODO: appending erases the tour structure. better keep them separated for
    # postprocessing, like counting how many times the particle gets to the top
    xtrace = A[]
    iptrace = Vector{I}[]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        i, nrst = take!(idle_nrst) # take an idle NRSTSampler. No other thread can now work on it
        xblk, ipblk = tour!(nrst)  # run a full tour with nrst
        append!(xtrace,xblk)
        append!(iptrace,ipblk)
        println("thread ", Threads.threadid(), ": ", "completed tour ", tour, " using sampler ", i)
        put!(idle_nrst, (i, nrst)) # mark nrst as idle
    end

    return xtrace, iptrace
end