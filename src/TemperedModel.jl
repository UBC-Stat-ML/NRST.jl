###############################################################################
# TemperedModel: encapsulates model specifics to allow for evaluating potentials
# and sampling from the reference
###############################################################################

abstract type TemperedModel end

Base.copy(tm::TemperedModel) = tm # default copy == don't copy anything

# compute both potentials. used within exploration kernels
function potentials(tm::TemperedModel, x)
    vref = Vref(tm, x)
    if isinf(vref) && vref > zero(vref) # x has 0 density under reference, V(x) is irrelevant
        return (vref, zero(vref))
    else
        return (vref, V(tm, x))
    end
end

# methods for sampling from the reference
function randrefwithv!(tm::TemperedModel, rng::AbstractRNG, x) # sample x∼π₀ and compute v=V(x)
    rand!(tm, rng, x) # sample new state from the reference
    return V(tm, x)   # return energy at new point
end
function randrefmayreject!(tm::TemperedModel, rng::AbstractRNG, x, reject_big_vs::Bool)
    v  = randrefwithv!(tm, rng, x)
    nv = 1
    while reject_big_vs && v > BIG
        nv += 1
        v   = randrefwithv!(tm, rng, x)
    end
    return v, nv
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
