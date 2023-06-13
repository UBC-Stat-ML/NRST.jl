##############################################################################
# methods for regenerative samplers
##############################################################################

abstract type RegenerativeSampler{T,TI<:Int,TF<:AbstractFloat} <: MCMCSampler{T,TI,TF} end

# run a full tour, starting from a renewal, and ending at the atom
# note: by doing the above, calling this function repeatedly should give the
# same output as the sequential version.
function tour!(rs::RegenerativeSampler, rng::AbstractRNG, tr::AbstractTrace; kwargs...)
    renew!(rs, rng)                           # init with a renewal
    while !isinatom(rs)
        save_pre_step!(rs, tr; kwargs...)     # save current state
        rp, xap, nvs = step!(rs, rng)         # do NRST step, produce new state, rej prob of temp step, and average xpl acc prob from expl step 
        save_post_step!(rs, tr, rp, xap, nvs) # save rej prob, xpl acc prob, and number of V(x) evals
    end
    save_last_step_tour!(rs, tr; kwargs...)
end

###############################################################################
# run in parallel exploiting regenerations
###############################################################################

# when called without trace_template, use a default one
function parallel_run(rs::RegenerativeSampler, rng::SplittableRandom; kwargs...)
    parallel_run(rs, rng, get_trace(rs); kwargs...)
end

# multithreading method
# uses a copy of rs with indep state per task. 
# note: rs itself is never used to sample so its state should be exactly the
# same after this returns
function parallel_run(
    rs::RegenerativeSampler,
    rng::SplittableRandom,
    trace_template::AbstractTrace;
    ntours::Int,
    verbose::Bool     = true,
    # check_every::Int  = 1_000,
    # max_gc_calls::Int = 3,
    # max_mem_use::Real = .8,
    kwargs...
    )
    @assert ntours>0
    GC.gc()                                                                   # setup allocates a lot so we need all mem we can get
    verbose && println(
        "\nRunning $ntours tours in parallel using " *
        "$(Threads.nthreads()) threads.\n"
    )
    
    # # detect and handle memory management within PBS
    # jobid= get_PBS_jobid()
    # ispbs= !(jobid == "")
    # mlim = ispbs ? get_cgroup_mem_limit(jobid) : Inf
    # ngcs = 0

    # pre-allocate traces and prngs, and then run in parallel
    res  = [similar(trace_template) for _ in 1:ntours]                        # get one empty trace for each task
    rngs = [split(rng) for _ in 1:ntours]                                     # split rng into ntours copies. must be done outside of loop because split changes rng state.
    p    = ProgressMeter.Progress(ntours; desc="Sampling: ", enabled=verbose) # prints a progress bar
    Threads.@threads for t in 1:ntours
        e = @elapsed tour!(copy(rs), rngs[t], res[t]; kwargs...)              # run a tour with tasks' own sampler, rng, and trace, avoiding race conditions. note: writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
        record_time(res[t], e)                                                # save elapsed time time
        ProgressMeter.next!(p)

        # # if on PBS, check every 'check_every' tours if mem usage is high. If so, gc.
        # if ispbs && mod(t, check_every)==0
        #     per_mem_used = get_cgroup_mem_usage(jobid)/mlim
        #     @debug "Tour $t: $(round(100*per_mem_used))% memory used."
        #     if per_mem_used > max_mem_use
        #         ngcs >= max_gc_calls && error("Reached max number of GC calls.")
        #         @debug "Calling GC.gc() due to usage above threshold"
        #         GC.gc()
        #         ngcs += 1
        #     end
        # end
    end
    TouringRunResults(res)                                                    # post-process and return 
end

# example output of the debug statements
# Sampling:  10%|████                                     |  ETA: 0:04:41┌ Debug: 76.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
# Sampling:  11%|████▌                                    |  ETA: 0:04:26┌ Debug: 92.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
# ┌ Debug: Calling GC.gc() due to usage above threshold
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:322
# Sampling:  11%|████▋                                    |  ETA: 0:07:30┌ Debug: 7.0% memory used.
# └ @ NRST /scratch/st-tdjc-1/mbironla/nrst-nextflow/work/conda/custom-conda-env-46669d40f024e9cb786ad8e9145aa8d1/share/julia/packages/NRST/H5HO4/src/NRSTSampler.jl:320
