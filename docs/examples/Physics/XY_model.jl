# ---
# cover: assets/XY_model.png
# title: XY model
# description: Simulating the classical XY lattice model.
# ---

# The model is the probability distribution of the angles ``\theta_s \in (-\pi,\pi)`` at each
# of the ``s\in\{1,\dots,S^2\}`` locations of a square lattice ``\mathbb{S}`` of side length
# equal to S. The tempered distributions have densities
# ```math
# \pi^{(\beta)}(\theta) = \frac{e^{-\beta V(\theta)}}{Z(\beta)}
# ```
# where
# ```math
# V(\theta) = -J\sum_{(a,b)\in\mathbb{S}} \cos(\theta_a-\theta_b)
# ```
# for ``J>0``. This potential is minimal for perfectly aligned configurations.
# Note that the reference distribution is uniform on ``(-\pi,\pi)^{S^2}``.

# For working with this model in the current implementation of NRST, we must 
# transform the state-space from ``(-\pi,\pi)^{S^2}`` to ``\mathbb{R}^{S^2}``.
# To do this we use the logistic-logit pair of transformations
# ```math
# \begin{aligned}
# x_s = T(\theta_s) &= \text{logit}\left(\frac{\theta_s}{2\pi} + \frac{1}{2}\right) \\
# \theta_s = T^{-1}(x_s) &= \pi(2\text{logistic}(x_s)-1)
# \end{aligned}
# ```
# Then, simulating from the transformed reference amounts to pushing samples of
# ``\theta`` through ``T(\cdot)``
# ```math
# \begin{aligned}
# \theta &\sim \text{U}(-\pi,\pi)^{S^2} \\
# x &= T(\theta)
# \end{aligned}
# ```
# It is not difficult to see that the reference density of the transformed variables
# ``x_s = T(\theta_s)`` is given by
# ```math
# \pi_x^{(0)}(x_s) = \text{logistic}(x_s)(1-\text{logistic}(x_s))
# ```
# In turn, this implies that the reference potential is
# ```math
# V_{\text{ref}}(x_s) := -\log(\pi_x^{(0)}(x_s)) = x + 2\log(1+\exp(-x))
# ```


# ## Implementation using NRST

using Lattices, LogExpFunctions, Distributions, Plots
using NRST

# Define the basics of the model
const S   = 8;
const Ssq = S*S;
const sq  = Square(S,S);
const βᶜ  = 1.1199;               # critical temp for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003
const J   = 2βᶜ;                  # coupling constant to force βᶜ = 0.5 in our parametrization
T(θ)      = logit((θ/pi + 1)/2)   # θ ∈ (-pi,pi) ↦ x ∈ ℝ
Tinv(x)   = pi*(2logistic(x) - 1) # x ∈ ℝ ↦ θ ∈ (-pi,pi)

# Define the potential function
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

# Define functions for the transformed reference
const dunif = Uniform(-pi,pi);
randref() = T.(rand(dunif, Ssq))
Vref(x::AbstractFloat) = x + 2log1pexp(-x) # = log1pexp(x) + log1pexp(-x)
Vref(xs::Vector{<:AbstractFloat}) = sum(Vref, xs)

# Build an NRST sampler for the model, tune it, sample with it, and show diagnostics
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

# save cover image #src
mkpath("assets") #src
savefig(plots[3], "assets/XY_model.png") #src

