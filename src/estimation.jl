###############################################################################
# utilities for estimating expectations of a test function h under level
# distributions, with calculation of standard errors
###############################################################################

# compute only the point estimate using a pre-processed xarray
# works with both SerialRunResults and ParallelRunResults
function point_estimate(
    res::RunResults;
    h::Function,                            # a real-valued test function defined on x space
    at::AbstractVector{<:Int} = [N(res)+1], # indexes ⊂ 1:(N+1) at which to estimate E^{i}[h(x)]
    agg::Function = mean                    # aggregation function
    )
    [isempty(xs) ? NaN64 : convert(Float64,agg(h.(xs))) for xs in res.xarray[at]]
end

# compute point estimate and its approximate asymptotic Monte Carlo variance
# using a ParallelRunResults
# if the sampler is repeatedly run independently for the same number of tours,
# 95% of the intervals means±1.96sqrt(vars/ntours) should contain the true posterior mean
function estimate(
    res::ParallelRunResults{T,TInt,TF};
    h::Function,                            # a real-valued test function defined on x space
    at::AbstractVector{<:Int} = [N(res)+1]  # indexes ⊂ 1:res.N at which to estimate E^{i}[h(x)]
    ) where {T,TInt,TF}
    means = point_estimate(res, h=h, at=at) # compute means using pre-processed res.xarray (fast)
    sumsq = zeros(TF, length(at))           # accumulate squared error accross tours
    tsum  = Vector{TF}(undef, length(at))   # temp for computing error within tour
    for tr in res.trvec
        fill!(tsum, zero(K)) # reset sums
        for (n, ip) in enumerate(tr.iptrace)
            # check if the index is in the requested set
            a = findfirst(isequal(ip[1] + 1), at)
            if !isnothing(a)
                tsum[a] += h(tr.xtrace[n]) - means[a]
            end
        end
        sumsq .+= tsum .* tsum
    end
    return (means = means, vars = sumsq ./ ntours(res))
end
