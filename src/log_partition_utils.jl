# trapezoidal approximation
# want: c[i] = c(x[i]) = int_0^x_i dx f(x)
# then c[0]=0, and for i>=1 use
# c[i] ≈ sum_{j=1}^i 0.5(f(x_j) + f(x_{j-1}))(x_j - x_{j-1})
# Or recursively,
# c[i] = c[i-1] + 0.5(f(x_i) + f(x_{i-1}))(x_i - x_{i-1})
function trapez!(c::T,xs::T,ys::T) where {K<:AbstractFloat,T<:Vector{K}}
    @assert length(c) == length(xs) == length(ys)
    c[1] = zero(K)
    oldx = xs[1]
    oldy = ys[1]
    for i in 2:length(xs)
        x = xs[i]
        y = ys[i]
        c[i] = c[i-1] + 0.5(y + oldy) * (x - oldx)
        oldx = x
        oldy = y
    end
end
function trapez(xs,ys)
    c = similar(ys)
    trapez!(c, xs, ys)
    return c
end


# stepping stone
# Z_N/Z_0 = prod_{i=1}^N Z_i/Z_{i-1}
# <=> log(Z_N/Z_0) = sum_{i=1}^N log(Z_i/Z_{i-1})
# Now,
# Z_i = E^{0}[e^{-beta_i V}] 
# = int pi_0(dx) e^{-beta_i V(x)}
# = int [pi_0(dx) e^{-beta_{i-1} V(x)}] e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} int pi^{i-1}(dx) e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# Hence
# Z_i/Z_{i-1} = E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# ≈ (1/S) sum_{n=1}^{S_{i-1}} e^{-(beta_i-beta_{i-1}) V(x_n)}, x_{1:S_{i-1}} ~ pi^{i-1}
# <=> log(Z_i/Z_{i-1}) ≈ -log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))
#  => log(Z_N/Z_0) = sum_{i=1}^N [-log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))]
# Recipe for the parallel version
# 1) samples in parallel V^{i}_{1:S_{i}}, for i ∈ 0:(N-1)
# 2) compute at each i ∈ (0,N-1): -log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))
# 3) cumsum
function stepping_stone!(
    zs::TV,
    bs::TV,
    trVs::Vector{TV}
    ) where {TF<:AbstractFloat, TV<:Vector{TF}}
    zs[1] = zero(TF)
    acc   = zero(TF)
    for i in 1:(length(zs)-1)
        db      = bs[i+1] - bs[i]
        acc    += logsumexp(-db*trVs[i]) - log(length(trVs[i]))
        zs[i+1] = acc
    end
end
function stepping_stone(bs, trVs)
    zs = similar(bs)
    stepping_stone!(zs, bs, trVs)
    return zs
end
