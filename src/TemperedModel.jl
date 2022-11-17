###############################################################################
# TemperedModel: encapsulates model specifics to allow for evaluating potentials
# and sampling from the reference
###############################################################################

abstract type TemperedModel end

potentials(tm::TemperedModel, x) = (Vref(tm,x), V(tm,x)) # compute and return both potentials
Base.copy(tm::TemperedModel) = tm                        # default copy == don't copy anything

# methods for sampling from the reference
function randrefwithv!(tm::TemperedModel, rng::AbstractRNG, x) # sample x∼π₀ and compute v=V(x)
    rand!(tm, rng, x) # sample new state from the reference
    return V(tm, x)   # return energy at new point
end
function randrefmayreject!(tm::TemperedModel, rng::AbstractRNG, x, reject_big_vs::Bool)
    v = randrefwithv!(tm, rng, x)
    while reject_big_vs && v > BIG
        v = randrefwithv!(tm, rng, x)
    end
    return v
end

# simple case: user passes proper Functions
# note: callable structs not allowed since they are probably modified when one
# evaluates them -> need special copy method
struct SimpleTemperedModel{TV<:Function,TVr<:Function,Tr<:Function} <: TemperedModel
    V::TV       # energy Function
    Vref::TVr   # energy of reference distribution
    randref::Tr # produces independent sample from reference distribution
end
Vref(tm::SimpleTemperedModel, x) = tm.Vref(x)
V(tm::SimpleTemperedModel, x) = tm.V(x)
Base.rand(tm::SimpleTemperedModel, rng::AbstractRNG) = tm.randref(rng)
