# ---
# cover: assets/XY_model.svg
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


# ## Implementation using NRST

using Lattices, Distributions, Plots
using Plots.PlotMeasures: px
using NRST

# Define the basics of the model
const S   = 8;
const Ssq = S*S;
const sq  = Square(S,S); # define a square lattice
const J   = 2;           # coupling constant to force βᶜ < 1 in our parametrization, since βᶜ = 1.1199 for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003

# Define the potential function
function V(θs::Vector{TF}) where {TF<:AbstractFloat}
    acc = zero(TF)
    for (a, b) in edges(sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end

# Define functions for the reference
const dunif = Uniform(-pi,pi)
randref() = rand(dunif, Ssq)
Vref(θ::AbstractFloat) = -logpdf(dunif, θ)
Vref(θs::Vector{<:AbstractFloat}) = sum(Vref, θs)

# This
# - builds an NRST sampler for the model
# - initializes it, finding an optimal grid
# - sample tours in parallel and uses them to get more accurate estimates of c(β)
# - sample one last time to show diagnostics
ns = NRSTSampler(
    V,
    Vref,
    randref,
    N = 12,
    verbose = true
)
res   = parallel_run(ns, ntours = 65_536)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
plot(plots..., layout = (hl,2), size = (900,hl*333),left_margin = 30px)

#md # ![Diagnostics plots](assets/XY_model_diags.svg)

# save diagnostics plot and cover image #src
mkpath("assets") #src
savefig("assets/XY_model_diags.svg") #src
savefig(plots[3], "assets/XY_model.svg") #src
