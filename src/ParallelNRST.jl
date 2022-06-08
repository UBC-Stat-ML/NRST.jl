###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# multithreading method
# note: ns itself is never used to sample so its state should be exactly the
# same after this returns
function parallel_run(
    ns::TS,
    rng::AbstractRNG;
    ntours::Int,
    keep_xs::Bool=true,
    kwargs...
    ) where {T,TI,TF,TS<:NRSTSampler{T,TI,TF}}
    println("\nRunning $ntours tours in parallel using $(Threads.nthreads()) threads.\n")
    nss  = [copy(ns) for _ in 1:ntours]                            # get one copy of ns per task. copying is fast relative to cost of a tour, and size(ns) ~ size(ns.x) 
    res  = [NRSTTrace(T,ns.np.N,TF,keep_xs) for _ in 1:ntours]     # get one empty trace for each task
    rngs = [split(rng) for _ in 1:ntours]
    p    = ProgressMeter.Progress(ntours, "Sampling: ")            # prints a progress bar
    Threads.@threads for t in 1:ntours
        tour!(nss[t], rngs[t], res[t]; keep_xs=keep_xs, kwargs...) # run a tour with the tasks' explorer and trace, avoiding race conditions. writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
        ProgressMeter.next!(p)
    end
    TouringRunResults(res)                                         # post-process and return 
end

