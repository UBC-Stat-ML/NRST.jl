# ###############################################################################
# # utilities for estimating expectations of a test function h under every
# # distribution, with calculation of standard errors
# ###############################################################################

function point_estimate(
    res::ParallelRunResults{T,I,K},
    h::Function,
    at::AbstractVector{<:Int},         # estimate E^{i}[h] for i in at
    aggfun::Function = Statistics.mean                  
    ) where {K,T,I}
    full_postprocessing!(res)
    [length(xs) > 0 ? aggfun(h.(xs)) : zero(K) for xs in res.xarray[at]]
end

function point_estimate(
    res::ParallelRunResults,
    h::Function,
    aggfun::Function = Statistics.mean
    )
    point_estimate(res, h, 1:res.N, aggfun)
end

# function estimate_se(
#     res::ParallelRunResults{T,I,K},
#     h::Function,
#     at:AbstractVector{<:Int}
#     ) where {K,T,I}
#     @unpack N, trace, ntours = res
#     tour = 0
#     for (_, xtrace, iptrace) in trace
#         tour += 1
#         len = 0
#         for (n, ip) in enumerate(iptrace)
#             len += 1
#             visits[tour, ip[1] + 1] += 1
#             push!(xarray[ip[1] + 1], xtrace[n])
#         end
#         nsteps[tour] = len
#         times[tour]  = seconds
#     end
# end


