module ExamplesGallery

import ..NRST: TemperedModel, V, Vref
import Base: rand

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
export MvNormalTM, free_energy, get_scaled_V_dist

end