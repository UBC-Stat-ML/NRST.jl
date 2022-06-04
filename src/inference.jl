###############################################################################
# utilities for estimating expectations of a test function h under level
# distributions, with calculation of standard errors
###############################################################################

# compute only the point estimate using a pre-processed xarray
# works with both SerialRunResults and ParallelRunResults
function point_estimate(
    res::RunResults;
    h::Function,                          # a real-valued test function defined on x space
    at::AbstractVector{<:Int} = [N(res)], # indexes ⊂ 0:N at which to estimate E^{i}[h(x)]
    agg::Function = mean                  # aggregation function
    )
    [isempty(xs) ? NaN64 : convert(Float64,agg(h.(xs))) for xs in res.xarray[at .+ 1]]
end

# using a ParallelRunResults, compute for a given real valued test function h and each level "at"
#   - point estimate
#   - asymptotic Monte Carlo variance of the point estimate
#   - posterior variance
#   - number of samples
#   - ESS(i) = nvisits(i)*(posterior variance)/(asymptotic variance)
# note: if the sampler is repeatedly run independently for the same number of tours,
# 95% of the intervals means±1.96sqrt(avars/ntours) should contain the true posterior mean
function inference(
    res::ParallelRunResults{T,TInt,TF};
    h,                                      # a real-valued test function defined on x space
    at::AbstractVector{<:Int} = [N(res)],   # indexes ⊂ 0:res.N at which to estimate E^{i}[h(x)]
    α::TF = 0.95                            # confidence level for asymptotic confidence intervals
    ) where {T,TInt,TF}
    means = point_estimate(res, h=h, at=at) # compute means using pre-processed res.xarray (fast)
    pvars = similar(means)                  # compute posterior variance
    for (p,i) in enumerate(at)
        pvars[p] = point_estimate(res, h=(x->abs2(h(x)-means[p])), at=[i])[1]
    end
    sumsq = zeros(TF, length(at))           # accumulate squared error accross tours
    tsum  = Vector{TF}(undef, length(at))   # temp for computing error within tour
    for tr in res.trvec
        fill!(tsum, zero(TF)) # reset sums
        for (n, ip) in enumerate(tr.trIP)
            # check if the index is in the requested set
            a = findfirst(isequal(ip[1]), at)
            if !isnothing(a)
                tsum[a] += h(tr.trX[n]) - means[a]
            end
        end
        sumsq .+= tsum .* tsum
    end
    avars = sumsq ./ ntours(res)            # compute asymptotic variance
    summarize_inference(res, at, α, means, avars, pvars)
end

# same as before but specialized for inferences on h(x) = h(V(x))
# works even for res objects that do not keep track of xs (i.e., keep_xs=false)
function inference_on_V(
    res::ParallelRunResults{T,TInt,TF};
    h,                                      # a real-valued test function defined on ℝ
    at::AbstractVector{<:Int} = [N(res)],   # indexes ⊂ 0:res.N at which to estimate E^{i}[h(V)]
    α::TF = 0.95                            # confidence level for asymptotic confidence intervals
    ) where {T,TInt,TF}
    # compute posterior means and variances
    means = Vector{TF}(undef, length(at))
    pvars = similar(means)                  
    for (p,i) in enumerate(at)
        hVs      = h.(res.trVs[i+1])
        m        = mean(hVs)
        v        = var(hVs, mean=m)
        means[p] = m
        pvars[p] = v
    end
    sumsq = zeros(TF, length(at))           # accumulate squared error accross tours
    tsum  = Vector{TF}(undef, length(at))   # temp for computing error within tour
    for tr in res.trvec
        fill!(tsum, zero(TF)) # reset sums
        for (n, ip) in enumerate(tr.trIP)
            # check if the index is in the requested set
            a = findfirst(isequal(ip[1]), at)
            if !isnothing(a)
                tsum[a] += h(tr.trV[n]) - means[a]
            end
        end
        sumsq .+= tsum .* tsum
    end
    avars = sumsq ./ ntours(res)            # compute asymptotic variance
    summarize_inference(res, at, α, means, avars, pvars)
end

# compute half width of α-CI, ESS, and build summarized dataframe
function summarize_inference(res::ParallelRunResults, at, α, means, avars, pvars)
    qmult    = quantile(Normal(), (1+α)/2)
    nsamples = vec(sum(res.visits[at .+ 1,:], dims=2))
    hws      = qmult * sqrt.(avars ./ nsamples) # half-widths of interval
    ESS      = nsamples .* (pvars ./ avars)
    return DataFrame(
        "Level"      => at,
        "Mean"       => means,
        "Asym. Var." => avars,
        "C.I. Low"   => means .- hws,
        "C.I. High"  => means .+ hws,
        "Post. Var." => pvars,
        "# Samples"  => nsamples,
        "ESS"        => ESS
    )
end

# estimate log-partition function: β ↦ log(Z(β)/Z(0))
function log_partition(np::NRSTProblem, res::RunResults)
    if np.use_mean
        return -np.c
    else
        return stepping_stone(np.betas, res.trVs)
    end
end

# TODO: revive this
# # estimate log-partition function, with a pesimistic asymptotic error bound
# # If I(n) is the cumulative integral approx at betas[n], then
# #    - sd(I(1)) = sd(mean(V[1]))
# #    - sd(I(n+1)) <= sd(I(n)) + sd(mean(V[n])) # when correlation = 1
# # Therefore, the upper and lower bounds can also be gotten using trapez!
# function log_partition(
#     ns::NRSTSampler{T,TInt,TF},
#     res::ParallelRunResults{T,TInt,TF};
#     α::TF = 0.95
#     ) where {T,TInt,TF}
#     # compute summary statistics for the potential function
#     # note: need Bonferroni adjustment because we need simultaneous coverage along
#     # the whole curve, not just marginally at every point.
#     @unpack fns, betas, N = ns.np
#     infres = inference(res, h = fns.V, at = 0:N, α = 1-(1-α)/N)
#     if infres[:,"Mean"][1] > 1e16
#         @info "log_partition: " *
#         "V likely not integrable under the reference; using stepping stone.\n" *
#         "Confidence region not yet implemented for this method."
#         trVs = [fns.V.(xs) for xs in res.xarray]
#         ms   = stepping_stone(betas, trVs)
#         lbs  = ubs = fill(convert(TF, NaN), N+1)     # no confidence region
#     else
#         ms   = -trapez(betas, infres[:,"Mean"])      # integrate the mean
#         lbs  = -trapez(betas, infres[:,"C.I. High"]) # integrate the upper bounds (Z decreases with V)
#         ubs  = -trapez(betas, infres[:,"C.I. Low"])  # integrate the lower bounds (Z decreases with V)
#     end
#     return (
#         lpart = DataFrame("Mean" => ms, "C.I. Low" => lbs, "C.I. High"  => ubs),
#         V     = infres
#     ) 
# end

