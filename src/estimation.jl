###############################################################################
# utilities for estimating expectations of a test function h under level
# distributions, with calculation of standard errors
###############################################################################

# compute only the point estimate
function point_estimate(
    res::ParallelRunResults{T,I,K},
    h::Function,
    at::AbstractVector{<:Int},      # indexes ⊂ 1:res.N at which to estimate E^{i}[h(x)]
    aggfun::Function = mean                  
    ) where {T,I,K}
    full_postprocessing!(res)
    [length(xs) > 0 ? aggfun(h.(xs)) : zero(K) for xs in res.xarray[at]]
end

# by default, compute h at all levels
function point_estimate(
    res::ParallelRunResults,
    h::Function,
    aggfun::Function = mean                  
    )
    point_estimate(res, h, 1:res.N, aggfun)
end

# compute point estimate and its approximate asymptotic Monte Carlo variance, so
# that ±1.96sqrt.(vars/res.ntours) gives approximately a 95% confidence coverage
function estimate(
    res::ParallelRunResults{T,I,K},
    h::Function,
    at::AbstractVector{<:Int} = 1:res.N # indexes ⊂ 1:res.N at which to estimate E^{i}[h(x)]
    ) where {T,I,K}
    means = point_estimate(res, h, at)
    sumsq = zeros(K, length(at))
    tsum  = Vector{K}(undef, length(at))
    for (_, xtrace, iptrace) in res.trace
        fill!(tsum, zero(K)) # reset sums
        for (n, ip) in enumerate(iptrace)
            a = findfirst(isequal(ip[1] + 1), at)
            if !isnothing(a)
                tsum[a] += h(xtrace[n]) - means[a]
            end
        end
        sumsq .+= tsum .* tsum
    end
    return (means = means, vars = sumsq ./ res.ntours)
end
