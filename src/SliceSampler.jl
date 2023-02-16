###############################################################################
# Slice sampler with doubling strategy (schema 4 in Neal (2003))
###############################################################################

struct SliceSampler{TTM<:TemperedModel,TF<:AbstractFloat,TV<:AbstractVector{TF},TI<:Int} <: ExplorationKernel
    # fields common to every ExplorationKernel
    tm::TTM                    # TemperedModel
    x::TV                      # current state (shared with NRSTSampler)
    curβ::Base.RefValue{TF}    # current beta
    curVref::Base.RefValue{TF} # current reference potential
    curV::Base.RefValue{TF}    # current target potential (shared with NRSTSampler)
    curVβ::Base.RefValue{TF}   # current tempered potential
    # idiosyncratic fields
    w::Base.RefValue{TF}       # initial window width
    p::TI                      # max window width = w2^p
end

# outer constructor
function SliceSampler(tm, x, curβ, curVref, curV, curVβ; w=10.0, p=20)
    SliceSampler(tm, x, curβ, curVref, curV, curVβ, Ref(w), p)
end
params(ss::SliceSampler) = (w=ss.w[],)                            # get params as namedtuple
function set_params!(ss::MHSampler, params::NamedTuple)           # set sigma from a NamedTuple
    ss.w[] = params.w
end