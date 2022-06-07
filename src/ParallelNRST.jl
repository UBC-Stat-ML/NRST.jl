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
    nss = [copy(ns) for _ in 1:ntours]                    # get one copy of ns per task. copying is fast relative to cost of a tour, and size(ns) ~ size(ns.x) 
    res = [NRSTTrace(T,ns.np.N,TF) for _ in 1:ntours]     # get one empty trace for each task
    p   = ProgressMeter.Progress(ntours, "Sampling: ")    # prints a progress bar
    @sync for t in 1:ntours                               # @sync tells the loop to end only when all @async operations inside end
        tsk = Threads.@spawn begin
            tour!(nss[t], res[t]; kwargs...)              # run a tour with the tasks' explorer and trace, avoiding race conditions. writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
            ProgressMeter.next!(p)
        end
        fetch(tsk)                                        # needed so that ProgressMeter works and also so that errors are thrown: https://discourse.julialang.org/t/i-dont-get-an-error-messages-after-a-task-thread-failed/36255/2
    end
    TouringRunResults(res)                                # post-process and return 
end

