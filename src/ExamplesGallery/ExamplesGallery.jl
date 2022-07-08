module ExamplesGallery

using UnPack: @unpack
import ..NRST: TemperedModel, V, Vref
import Base: rand

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
export MvNormalTM, free_energy, get_scaled_V_dist

end