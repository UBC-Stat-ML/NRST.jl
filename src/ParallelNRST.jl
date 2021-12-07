###############################################################################
# run NRST in parallel exploiting regenerations
# uses the multithreaded approach described here
# https://discourse.julialang.org/t/request-for-comments-on-approach-to-multithreaded-work/72728?u=miguelbiron
###############################################################################

function parallel_run!(
    ns::NRSTSampler;                 # (must have already been initially-tuned)
    ntours::Int = 20,                # run ntours independent tours
    nthrds::Int = Threads.nthreads() # number of threads available to run tours
)
    # channel containing idle nrst samplers, which can be used by workers
    # initialize with ns and additional (nthrds-1) deepcopies 
    idle_nrst = Channel{Tuple{Int,NRSTSampler}}(nthrds)
    put!(idle_nrst, (1, ns))
    foreach(i -> put!(idle_nrst, (i, NRSTSampler(ns))), 2:nthrds)

    # do jobs using nthrds workers, taking and putting back idle resources
    Threads.@threads for tour in 1:ntours
        i, nrst = take!(idle_nrst) # take an idle NRSTSampler. No other thread can now work on it
        xtrace, iptrace = tour!(ns)
        println("thread ", threadid(), ": ", "completed tour ", tour, " using sampler ", i)
        put!(idle_res, res) # mark resource as idle
    end
end