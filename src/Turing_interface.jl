###############################################################################
# Minimal interface that lets one obtain randref, Vref and V from a DynamicPPL.Model
# The transformation 𝕏 → ℝ from constrained to unconstrained Euclidean space
# is used to simplify the work for exploration kernels 
###############################################################################

# storage for functions associated to a tempered problem
struct TuringFuns{TV,TVr,Tr,Tmod,Tspl,TVi} <: Funs
    V::TV       # energy Function
    Vref::TVr   # energy of reference distribution
    randref::Tr # produces independent sample from reference distribution
    model::Tmod # a DynamicPPL.Model
    spl::Tspl   # a DynamicPPL.Sampler
    viout::TVi  # a DynamicPPL.VarInfo
end

function TuringFuns(model::Model)
    viout   = DynamicPPL.VarInfo(model)             # build a TypedVarInfo
    spl     = DynamicPPL.Sampler(Turing.HMC(0.1,5)) # build a dummy sampler that works in unconstrained space
    DynamicPPL.link!(viout, spl)                    # force transformation 𝕏 → ℝ
    randref = gen_randref(viout, spl, model)
    V       = gen_V(viout, spl, model)
    Vref    = gen_Vref(viout, spl, model)
    TuringFuns(V, Vref, randref, model, spl, viout)
end

# copy a TuringFuns. it keeps model and spl common. Avoiding copying model is
# especially important because it contains the dataset, which can be huge
function Base.copy(fns::TuringFuns)
    @unpack model, spl = fns
    vinew = DynamicPPL.VarInfo(model)              # build a new TypedVarInfo
    DynamicPPL.link!(vinew, spl)                   # link with old sampler to force transformation 𝕏 → ℝ
    randref = gen_randref(vinew, spl, model)
    V       = gen_V(vinew, spl, model)
    Vref    = gen_Vref(vinew, spl, model)
    TuringFuns(V, Vref, randref, model, spl, vinew)
end

# # TODO: when the MiniBatchContext gets fixed, use it to build custom gen_Vβ
# function gen_Vβ(fns::TuringFuns, ind::Int, betas::AbstractVector{<:AbstractFloat})
#     ...
# end

# NRSTSampler constructor
function NRSTSampler(model::Model, args...;kwargs...)
    fns = TuringFuns(model)
    NRSTSampler(fns, args...;kwargs...)
end

#######################################
# utilities
#######################################

# generate a closure to get a (transformed!) iid sample from the prior
# TODO: it might be more efficient to create (and link!) vi once outside the inner fn,
# so that it is captured in the closure, and then just use model(). However,
# I haven't been able to "reuse" a typed VarInfo for sampling
function gen_randref(::Tvi, spl, model) where {Tvi} # passing the type of viout is enough to make randref type-stable
    function randref(rng::AbstractRNG=Random.GLOBAL_RNG)
        vi::Tvi = DynamicPPL.VarInfo(rng, model, SampleFromPrior(), PriorContext()) # avoids evaluating the likelihood
        DynamicPPL.link!(vi, spl)
        vi[spl]
    end
    return randref
end

# generate functions to compute prior (Vref) and likelihood (V) potentials
# these functions act on transfomed (i.e., unconstrained) variables
# simplified and modified version of gen_logπ in Turing
# https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L444
# difference: it does not retain the original value in vi, which we don't really need
# Note: can use hasproperty(V,:vi) to distinguish V's created with Turing interface
# TODO: this is inefficient because it requires 2 passes over the graph to get Vref + βV,
# but this would probably need improving the way we work with Vref and V in NRST
function gen_Vref(viout, spl, model)
    function Vref(x)::Float64
        vi  = viout # this helps with the re-binding+Boxing issue: https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
        vi  = DynamicPPL.setindex!!(vi, x, spl)
        vi  = last(DynamicPPL.evaluate!!(model, vi, spl, PriorContext()))
        pot = -getlogp(vi)
        return pot
    end
    return Vref
end
function gen_V(viout, spl, model)
    function V(x)::Float64
        vi  = viout # this helps with the re-binding+Boxing issue: https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
        vi  = DynamicPPL.setindex!!(vi, x, spl)
        vi  = last(DynamicPPL.evaluate!!(model, vi, spl, LikelihoodContext()))
        pot = -getlogp(vi)
        return pot
    end
    return V
end

###############################################################################
# TODOs
###############################################################################

# ####################################################
# # evaluating tempered posteriors
# # does NOT work when link!'d
# ####################################################

# β = 0.7
# vi = VarInfo(rng, lnmodel)
# s = vi[@varname(s)]
# getlogp(vi) ≈ logpdf(LogNormal(),s) + sum(logpdf.(Normal(0.,s),lnmodel.args[1]))
# mbctx = MiniBatchContext(DefaultContext(),β)
# vi = last(DynamicPPL.evaluate!!(lnmodel, vi, mbctx))
# getlogp(vi) ≈ logpdf(LogNormal(),s) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

# link!(vi, spl)
# istrans(vi,@varname(s))
# vi = last(DynamicPPL.evaluate!!(lnmodel, rng, vi, spl, mbctx)) # since mbctx uses DefaultContext, this does not sample, only evaluates
# theta = vi[spl]
# theta[1] == log(s)
# getlogp(vi) ≈ (logpdf(LogNormal(),s) + log(s)) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

# using NRST

# # this works :)
# V    = gen_V(vi, spl, lnmodel)
# Vref = gen_Vref(vi, spl, lnmodel)
# -(Vref(theta) + β*V(theta)) ≈ (logpdf(LogNormal(),s) + log(s)) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

# ####################################################
# # step with HMC
# ####################################################

# # initial step
# t, state = DynamicPPL.AbstractMCMC.step(rng, lnmodel, spl)
# Turing.Inference.getparams(t)
# Turing.Inference.getparams(state.vi)
# Turing.Inference.metadata(t)

# s = state.vi[@varname(s)]
# state.vi[spl][1] == log(s)
# getlogp(state.vi) == state.hamiltonian.ℓπ(state.vi[spl])

# # subsequent steps
# _, state = Turing.AbstractMCMC.step(rng, lnmodel, spl, state)

# ####################################################
# # invlink using DPPL machinery
# ####################################################

# vns = DynamicPPL._getvns(ns.np.V.viout,ns.np.V.spl)
# typeof(vns)
# typeof(vns[1])
# dist = DynamicPPL.getdist(ns.np.V.viout, vns[1][1])
# DynamicPPL.Bijectors.invlink(
#     dist,
#     reconstruct(
#         dist,
#         DynamicPPL.getval(ns.np.V.viout, vns[1])
#     )
# )

