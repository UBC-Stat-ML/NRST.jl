using Lattices: Square, edges
using IrrationalConstants: twoπ, log2π

abstract type LatticeTemperedModel <: TemperedModel end

struct XYModel{TF<:AbstractFloat, TSq<:Square, TI<:Int} <: LatticeTemperedModel
    Sq::TSq
    S::TI
    S²::TI
    J::TF     # coupling constant to force βᶜ < 1 in our parametrization, since βᶜ = 1.1199 for J=1: https://iopscience.iop.org/article/10.1088/0305-4470/38/26/003
    Vref0::TF
end
XYModel(S::Int, J::AbstractFloat=2.) = XYModel(Square(S,S), S, S*S, J, S*S*log2π)

# Define the potential function
function V(tm::XYModel{TF}, θs::Vector{TF}) where {TF<:AbstractFloat}
    @unpack Sq, S, J = tm
    acc = zero(TF)
    for (a, b) in edges(Sq)
        ia   = (a[1]-1)*S + a[2]
        ib   = (b[1]-1)*S + b[2]
        acc -= cos(θs[ia] - θs[ib])
    end
    return J*acc
end

# Define functions for the reference
Base.rand(tm::XYModel, rng) = (twoπ*rand(rng, tm.S²) .- pi)
Vref(tm::XYModel, θs::Vector{<:AbstractFloat}) = 
    any(θ -> (θ <= -pi || θ > pi), θs) ? Inf : tm.Vref0
