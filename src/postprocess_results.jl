###############################################################################
# postprocessing ParallelRunResults
###############################################################################

function tour_durations!(res::ParallelRunResults{T,I,K}) where {T,I,K}
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

function full_postprocessing!(res::ParallelRunResults)
    if res.status[] >= 3
        return
    end
    @unpack trace, N, nsteps, ntours, times, xarray, visits, toureff = res
    sumvs = zeros(Int, N)
    sumsq = zeros(Int, N)
    tsum  = Vector{Int}(undef, N)
    tour  = 0
    for (seconds, xtrace, iptrace) in trace      # iterate tours
        tour += 1
        fill!(tsum, zero(Int))                   # reset tour sums
        for (n, ip) in enumerate(iptrace)
            idx = ip[1] + 1                      # switch to 1-based indexing
            tsum[idx]         += 1
            visits[tour, idx] += 1
            push!(xarray[idx], xtrace[n])        # add current x to the array for the idx level
        end
        if res.status[] < 2
            nsteps[tour] = length(iptrace)
            times[tour]  = seconds
        end
        sumvs .+= tsum
        sumsq .+= (tsum .* tsum)
    end
    # compute tour effectiveness
    for i in 1:N
        mv         = sumvs[i] / ntours
        toureff[i] = mv * mv / (sumsq[i]/ntours) 
    end
    res.status[] = 3
    return
end