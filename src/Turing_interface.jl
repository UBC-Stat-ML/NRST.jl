###############################################################################
# Minimal interface that lets one obtain randref, Vref and V from a DynamicPPL.Model
# The transformation 𝕏 → ℝ from constrained to unconstrained Euclidean space
# is used to simplify the work for exploration kernels 
###############################################################################

# NRSTSampler constructor
function NRSTSampler(model::Model, betas, nexpl, use_mean)
    # build a TypedVarInfo and a dummy sampler that forces 𝕏 → ℝ trans via `link!`
    viout   = DynamicPPL.VarInfo(model)
    spl     = DynamicPPL.Sampler(Turing.HMC(0.1,5))
    DynamicPPL.link!(viout, spl)
    randref = gen_randref(viout, spl, model)
    V       = gen_V(viout, spl, model)
    Vref    = gen_Vref(viout, spl, model)
    NRSTSampler(V, Vref, randref, betas, nexpl, use_mean)
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

