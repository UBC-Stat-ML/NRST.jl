###############################################################################
# TemperedModel: encapsulates model specifics to allow for evaluating potentials
# and sampling from the reference
###############################################################################

abstract type TemperedModel end

# compute and return both potentials
potentials(tm::TemperedModel, x) = (Vref(tm,x), V(tm,x))

# simple case: user passes proper Functions
# note: callable structs not allowed since they probably are modified by evaluating V's
struct SimpleTemperedModel{TV<:Function,TVr<:Function,Tr<:Function} <: TemperedModel
    V::TV       # energy Function
    Vref::TVr   # energy of reference distribution
    randref::Tr # produces independent sample from reference distribution
end
Vref(tm::SimpleTemperedModel, x) = tm.Vref(x)
V(tm::SimpleTemperedModel, x) = tm.V(x)
Base.rand(tm::SimpleTemperedModel) = tm.randref()
Base.copy(tm::SimpleTemperedModel) = tm # dont do anything
