###############################################################################
# postprocessing ParallelRunResults
###############################################################################

function tour_durations!(res::ParallelRunResults{T,I,K}) where {K,T,I}
    if res.status[] >= 2
        return
    end
    @unpack trace, nsteps, times = res
    tour = 0
    for (seconds, _, iptrace) in trace
        tour += 1
        nsteps[tour] = length(iptrace)
        times[tour]  = seconds
    end
    res.status[] = 2
    return
end

function full_postprocessing!(res::ParallelRunResults{T,I,K}) where {K,T,I}
    if res.status[] >= 3
        return
    end
    @unpack trace, nsteps, times, xarray, visits = res
    tour = 0
    for (seconds, xtrace, iptrace) in trace
        tour += 1
        for (n, ip) in enumerate(iptrace)
            visits[tour, ip[1] + 1] += 1
            push!(xarray[ip[1] + 1], xtrace[n])
        end
        if res.status[] < 2
            nsteps[tour] = length(iptrace)
            times[tour]  = seconds
        end
    end
    res.status[] = 3
    return
end