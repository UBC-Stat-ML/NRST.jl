using Lattices, LogExpFunctions, Distributions, Plots
using NRST

const S   = 16;
const Ssq = S*S;
const sq  = Square(S,S);
const βᶜ  = 1.1199 # critical temp for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003
const J   = 2βᶜ    # coupling constant to force βᶜ = 0.5 in our parametrization 
T(θ)      = logit((θ/pi + 1)/2)   # θ ∈ (-pi,pi) ↦ x ∈ ℝ
Tinv(x)   = pi*(2logistic(x) - 1) # x ∈ ℝ ↦ θ ∈ (-pi,pi)
function V(xs::Vector{TF}) where {TF<:AbstractFloat}
    acc = zero(TF)
    θs  = Tinv.(xs)
    for (a, b) in edges(sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end
# V(10randn(Ssq)) > V(fill(rand(),Ssq)) # minimal under perfect alignment

# prior = uniform on (-pi,pi)^Ssq
const dunif = Uniform(-pi,pi);
randref() = T.(rand(dunif, Ssq))
# histogram(Tinv.(vcat([randref() for _ in 1:100]...)),bins=50) # should be ~ U(-pi,pi)

# prior on x induced by θ ~ U(-pi,pi) and x = T(θ)
# pX(x) = pθ(T^{-1}(x)) |dT^{-1}(x)/dx| = (2pi)^{-1} [|dT/dθ|^{-1}](T^{-1}(x))
# now
# dlogit(z)/dz = (z(1-z))^{-1}
# while
# z(θ) = (θ/π + 1)/2 => dz/dθ = (2π)^{-1}
# so
# dT/dθ = [dlogit(z)/dz][dz/dθ] = {z(θ)[1-z(θ)]}^{-1} (2π)^{-1}
# hence
# [|dT/dθ|^{-1}](T^{-1}(x)) = 2π z(θ(x))[1-z(θ(x))]
# finally
# pX(x) = (2pi)^{-1} [|dT/dθ|^{-1}](T^{-1}(x)) = z(θ(x))[1-z(θ(x))]
# now
# θ(x) = π(2logistic(x) - 1) 
# => z(x) = (θ(x)/π + 1)/2 = (2logistic(x) - 1 + 1)/2 = logistic(x)
# Hence
# pX(x) = logistic(x)[1-logistic(x)] = [1/(1+exp(-x))][exp(-x)/(1+exp(-x))]
# = exp(-x)(1+exp(-x))^{-2}
# <=> log(pX(x)) = -x -2log(1+exp(-x)) = -x - 2log1pexp(-x) = -(x + 2log1pexp(-x))
# OTOH, 1-logistic(x) = logistic(-x), so
# pX(x) = logistic(x)logistic(-x)
# = [1/(1+exp(-x))][1/(1+exp(x))]
# <=> log(pX(x)) = -(log(1+exp(-x)) + log(1+exp(x)))
# = -(log1pexp(x) + log1pexp(-x))
# same but more expensive due to the 2 calls to log1pexp
Vref(x::AbstractFloat) = x + 2log1pexp(-x) # = log1pexp(x) + log1pexp(-x)
Vref(xs::Vector{<:AbstractFloat}) = sum(Vref, xs)

# # check using Bijectors
# using Bijectors
# all([Vref(x) ≈ -logpdf_with_trans(dunif,Tinv(x),true) for x in randn(100)])

# # check that pX() is normalized
# xs = range(-10,10,1000)
# vs = [Vref(x) for x in xs]
# lΔ = log(step(xs))
# abs(logsumexp(lΔ .- vs)) < 1e-4

# sample
ns = NRSTSampler(
    V,
    Vref,
    randref,
    N = 200,
    verbose = true
);
res = parallel_run(ns, ntours = 1024);
plots = diagnostics(ns,res);
plot(plots..., layout = (3,2), size = (800,1000))
