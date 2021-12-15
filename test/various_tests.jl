M = Matrix{Int}(undef, 2, 3)
v = [7, 3]
using StaticArrays

cpy!(M,v,n) = copyto!(M, 1:2, n:n, v, 1:2, 1:1)
sv = MVector{2,Int}(3,15)
@time cpy!(M,sv,3)
copyto!(M, 1:2, 1:1, MVector{2,Int}(0,1), 1:2, 1:1)
copyto!(M, CartesianIndices((1:2,3:3)), v, CartesianIndices(v))
