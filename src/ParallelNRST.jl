###############################################################################
# run NRST in parallel exploiting regenerations
###############################################################################

# create vector of identical copies of a given nrst sampler
# used to avoid race conditions between threads
# initialize with ns and additional nthrds-1 (deep)copies
function get_samplers_vector(
    ns::NRSTSampler{T,I,K,B};        
    nthrds::Int
) where {T,I,K,B}
    samplers = Vector{NRSTSampler{T,I,K,B}}(undef, nthrds)
    samplers[1] = ns
    for i in 2:nthrds
        samplers[i] = NRSTSampler(ns)
    end
    return samplers
end

# utility that creates the samplers vector for you and then uses it to run
function parallel_run!(
    ns::NRSTSampler;        
    ntours::Int,                      # run a total of ntours
    nthrds::Int = Threads.nthreads(), # number of threads available to run tours
    verbose::Bool = false
)
    samplers = get_samplers_vector(ns, nthrds = nthrds)
    trace = parallel_run!(samplers, ntours=ntours, verbose=verbose)
    return (samplers = samplers, trace = trace)
end

function parallel_run!(
    samplers::Vector{NRSTSampler{T,I,K,B}}; # contains (deep)copies of an NRSTSampler object 
    ntours::Int,                            # run a total of ntours tours
    verbose::Bool = false
) where {T,I,K,B}
    # need separate storage for each threadid because push! is not atomic!
    trace_vec = [Tuple{K, Vector{T}, Vector{MVector{2,I}}}[] for _ in 1:length(samplers)]

    # run tours in parallel using the available nrst's in the channel
    Threads.@threads for tour in 1:ntours
        seconds = @elapsed begin
            xtrace, iptrace = tour!(samplers[Threads.threadid()]) # run a full tour with the sampler corresponding to this thread, avoiding race conditions
        end
        push!(
            trace_vec[Threads.threadid()],                        # push results to this thread's own storage, avoiding race condition
            (seconds, xtrace, iptrace)
        )
        verbose && println(
            "thread ", Threads.threadid(), ": completed tour ", tour
        )
    end
    
    trace = vcat(trace_vec...)
    N = length(samplers[1].np.betas)
    return postprocess_tours(ntours, N, trace)
end

# process the raw trace
function postprocess_tours(
    ntours::Int,
    N::Int,
    trace::Vector{Tuple{K, Vector{T}, Vector{MVector{2,I}}}}
    ) where {K,T,I}

    xarray = [T[] for _ in 1:N]       # sequentially collect the value of x at each of the levels
    nsteps = Vector{I}(undef, ntours) # number of steps in every tour
    times  = Vector{K}(undef, ntours) # time spent in every tour
    visits = zeros(I, ntours, N)      # number of visits to each level on each tour
    iptracefull = zeros(I, 2 ,0)      # collect the whole index process in one big matrix (for the obligatory plot)
    tour = 0
    for (seconds, xtrace, iptrace) in trace
        tour += 1
        len = 0
        for (n, ip) in enumerate(iptrace)
            len += 1
            visits[tour, ip[1] + 1] += 1
            push!(xarray[ip[1] + 1], xtrace[n])
        end
        nsteps[tour] = len
        times[tour]  = seconds
        iptracefull  = hcat(iptracefull, reduce(hcat, iptrace))
    end
    @assert tour == ntours
    return (
        nsteps = nsteps, times = times, xarray = xarray, visits = visits,
        ip = collect(iptracefull')
    )
end
