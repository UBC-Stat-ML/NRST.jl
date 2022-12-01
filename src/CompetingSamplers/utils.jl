# sample from log-probs
# adapted from StatsBase method: https://github.com/JuliaStats/StatsBase.jl/blob/bd4ca61f4bb75f2c6cd0a47aee1cfde7b696eb9c/src/sampling.jl#L552
# when we have ps = exp.(lps), then sampling does 
#     U ~ U[0,1]
#     n(U) = min{n in 1:m : sum_{j=1}^n ps[j] >= U }
# but 
#     sum_{j=1}^n ps[j] >= U <=> log(sum_{j=1}^n ps[j]) >= log(U)
# Hence, we can sample from lps via
#     E ~ Exp(1)
#     n(E) = min{n in 1:m : logsumexp(ps[1:n]) >= -E } 
function sample_logprob(rng::AbstractRNG, lps::AbstractVector)
    nE  = -randexp(rng)
    M   = length(lps)
    m   = 1
    clp = lps[1]
    while clp < nE && m < M
        m += 1
        @inbounds clp = logaddexp(clp, lps[m])
    end
    return m
end
