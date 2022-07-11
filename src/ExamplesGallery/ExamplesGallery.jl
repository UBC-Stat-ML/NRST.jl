module ExamplesGallery

using DelimitedFiles: readdlm
using Distributions
using DynamicPPL
using IrrationalConstants: twoπ, log2π
using Lattices: Square, edges
using UnPack: @unpack
import ..NRST: NRST, TemperedModel, TuringTemperedModel, V, Vref
import Base: rand

# Turing
include("Turing/hierarchical_model.jl")
export HierarchicalModel

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
export MvNormalTM, free_energy, get_scaled_V_dist

end