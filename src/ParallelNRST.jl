###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

"""
    ParallelRunResults

A struct containing raw results as well as postprocessed quantities

- `:nsteps`: vector of length `ntours` containing the total number of
             steps of each tour
- `:times` : vector of length `ntours` containing the total time spent
             in each tour
- `:xarray`: vector of length `N+1`. The `i`-th entry contains samples from the
             `i`-th annealed distribution, and its length is equal to the total
             number of visits to that level.
- `:visits`: matrix of size `ntours × N` containing the number of visits
             to each state in each tour
"""
struct ParallelRunResults{T,I,K}
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}}
    N::Int
    ntours::Int
    nthrds::Int
    nsteps::Vector{I}         # number of steps in every tour
    times::Vector{K}          # time spent in every tour
    xarray::Vector{Vector{T}} # sequentially collect the value of x at each of the levels
    visits::Matrix{I}         # number of visits to each level on each tour
    status::Base.RefValue{Int}# 1: first 4 fields computed // 2: first 6 // 3: all
end

# outer constructor for minimal initialization (status 1)
function ParallelRunResults(
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}},
    N::Int,
    ntours::Int,
    nthrds::Int
) where {T,I,K}
    nsteps = Vector{I}(undef, ntours)
    times  = Vector{K}(undef, ntours)
    xarray = [T[] for i in 1:N]
    visits = zeros(I, ntours, N)
    ParallelRunResults(
        trace, N, ntours, nthrds, nsteps, times, xarray, visits, Ref(1)
    )
end

# create vector of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthrds-1 (deep)copies
# note that np object is still shared since it's not modified during simulation
function copy_sampler(
    ns::NRSTSampler{T,I,K,B};        
    nthrds::Int
) where {T,I,K,B}
    samplers = Vector{NRSTSampler{T,I,K,B}}(undef, nthrds)
    samplers[1] = ns
    for i in 2:nthrds
        samplers[i] = NRSTSampler(ns) # copy constructor
    end
    return samplers
end

# run in parallel using a vector of identical copies of NRST samplers
function parallel_run!(
    samplers::Vector{NRSTSampler{T,I,K,B}}; # contains nthrds (deep)copies of an NRSTSampler object 
    ntours::Int,                            # run a total of ntours tours
    verbose::Bool = false
) where {T,I,K,B}
    # need separate storage for each threadid because push! is not atomic!
    trace_vec = [Tuple{K, Vector{T}, Vector{MVector{2,I}}}[] for _ in 1:length(samplers)]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        tstats = @timed begin
            xtrace, iptrace = tour!(samplers[Threads.threadid()]) # run a full tour with the sampler corresponding to this thread, avoiding race conditions
        end
        push!(
            trace_vec[Threads.threadid()],                        # push results to this thread's own storage, avoiding race condition
            (tstats.time - tstats.gctime, xtrace, iptrace)        # remove GC time from total elapsed time
        )
        verbose && println(
            "thread ", Threads.threadid(), ": completed tour ", tour
        )
    end
    
    trace = vcat(trace_vec...)
    ParallelRunResults(
        trace, length(samplers[1].np.betas), ntours, length(samplers)
    )
end
