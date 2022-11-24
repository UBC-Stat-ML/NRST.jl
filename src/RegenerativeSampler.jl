#######################################
# touring interface
#######################################

abstract type RegenerativeSampler{T,TI<:Int,TF<:AbstractFloat,TXp<:ExplorationKernel,TProb<:NRSTProblem} end

# method that allocates a trace object
function tour!(rs::RegenerativeSampler{T,I,K}, rng::AbstractRNG; kwargs...) where {T,I,K}
    tr = NRSTTrace(T, rs.np.N, K)
    tour!(rs, rng, tr; kwargs...)
    return tr
end

# run a full tour, starting from a renewal, and ending at the atom
# note: by doing the above, calling this function repeatedly should give the
# same output as the sequential version.
function tour!(
    rs::RegenerativeSampler{T,I,K},
    rng::AbstractRNG, 
    tr::NRSTTrace; 
    kwargs...
    ) where {T,I,K}
    renew!(rs, rng)                         # init with a renewal
    while !isinatom(rs)
        save_pre_step!(rs, tr; kwargs...)   # save current state
        rp, xplap = step!(rs, rng)          # do NRST step, produce new state, rej prob of temp step, and average xpl acc prob from expl step 
        save_post_step!(rs, tr, rp, xplap)  # save rej prob and xpl acc prob
    end
    save_pre_step!(rs, tr; kwargs...)       # store state at atom
    save_post_step!(rs, tr, one(K), K(NaN)) # we know that (-1,-1) would be rejected if attempted so we store this. also, the expl step would not use an explorer; thus the NaN.
end

# run multiple tours (serially), return processed output
function run_tours!(
    rs::RegenerativeSampler{T,TI,TF},
    rng::AbstractRNG;
    ntours::Int,
    kwargs...
    ) where {T,TI,TF}
    results = Vector{NRSTTrace{T,TI,TF}}(undef, ntours)
    ProgressMeter.@showprogress 1 "Sampling: " for t in 1:ntours
        results[t] = tour!(rs, rng;kwargs...)
    end
    return TouringRunResults(results)
end

###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# multithreading method
# uses a copy of rs with indep state per task. 
# note: rs itself is never used to sample so its state should be exactly the
# same after this returns
function parallel_run(
    rs::TS,
    rng::SplittableRandom,
    ntours::Int;
    keep_xs::Bool     = true,
    verbose::Bool     = true,
    check_every::Int  = 1_000,
    max_mem_use::Real = .8,
    kwargs...
    ) where {T,TI,TF,TS<:RegenerativeSampler{T,TI,TF}}
    GC.gc()                                                                   # setup allocates a lot so we need all mem we can get
    verbose && println(
        "\nRunning $ntours tours in parallel using " *
        "$(Threads.nthreads()) threads.\n"
    )
    
    # detect and handle memory management within PBS
    jobid= get_PBS_jobid()
    ispbs= !(jobid == "")
    mlim = ispbs ? get_cgroup_mem_limit(jobid) : TF(Inf)

    # pre-allocate traces and prngs, and then run in parallel
    res  = [NRSTTrace(T,rs.np.N,TF) for _ in 1:ntours]                        # get one empty trace for each task
    rngs = [split(rng) for _ in 1:ntours]                                     # split rng into ntours copies. must be done outside of loop because split changes rng state.
    p    = ProgressMeter.Progress(ntours; desc="Sampling: ", enabled=verbose) # prints a progress bar
    Threads.@threads for t in 1:ntours
        tour!(copy(rs), rngs[t], res[t]; keep_xs=keep_xs, kwargs...)          # run a tour with tasks' own sampler, rng, and trace, avoiding race conditions. note: writing to separate locations in a common vector is fine. see: https://discourse.julialang.org/t/safe-loop-with-push-multi-threading/41892/6, and e.g. https://stackoverflow.com/a/8978397/5443023
        ProgressMeter.next!(p)

        # if on PBS, check every 'check_every' tours if mem usage is high. If so, gc.
        if ispbs && mod(t, check_every)==0
            per_mem_used = get_cgroup_mem_usage(jobid)/mlim
            @debug "Tour $t: $(round(100*per_mem_used))% memory used."
            if per_mem_used > max_mem_use
                @debug "Calling GC.gc() due to usage above threshold"
                GC.gc()
            end
        end
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
