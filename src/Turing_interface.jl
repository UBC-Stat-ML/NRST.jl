###############################################################################
# Minimal interface to obtain potentials and sampling from the prior
# The transformation 𝕏 → ℝ from constrained to unconstrained Euclidean space
# is used to simplify the work for exploration kernels 
###############################################################################

const DPPL = DynamicPPL

# TemperedModel built from a Turing model
# struct TuringTemperedModel{TV,TVr,Tr,Tmod,Tspl,TVi} <: TemperedModel
struct TuringTemperedModel{Tm<:DPPL.Model,Ts<:DPPL.AbstractSampler,TVi<:DPPL.AbstractVarInfo} <: TemperedModel
    model::Tm  # a DPPL.Model
    spl::Ts    # a DPPL.Sampler
    viout::TVi # a DPPL.VarInfo
end

# outer constructor
function TuringTemperedModel(model::DPPL.Model)
    viout   = DPPL.VarInfo(model)            # build a TypedVarInfo
    spl     = DPPL.SampleFromPrior()         # used for sampling and to "link!" (transform to unrestricted space)
    DPPL.link!(viout, spl)                   # force transformation 𝕏 → ℝ
    TuringTemperedModel(model, spl, viout)
end

# copy a TuringTemperedModel. it keeps model.args common because that is the
# data, which can be huge
function Base.copy(tm::TuringTemperedModel)
    newmodel = tm.model.f(tm.model.args...)
    TuringTemperedModel(newmodel)
end

# NRSTSampler constructor
function NRSTSampler(model::DPPL.Model, args...;kwargs...)
    tm = TuringTemperedModel(model)
    NRSTSampler(tm, args...;kwargs...)
end

#######################################
# utilities
#######################################

#######################################
# methods
#######################################

# sampling from the prior
function Base.rand(tm::TuringTemperedModel, rng::AbstractRNG)
    vi = DPPL.VarInfo(rng, tm.model, tm.spl, DPPL.PriorContext()) # avoids evaluating the likelihood
    DPPL.link!(vi, tm.spl)
    vi[tm.spl]
end

# evaluate reference potential
function Vref(tm::TuringTemperedModel, x)
    vi  = tm.viout                            # this helps with the re-binding+Boxing issue: https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
    vi  = DPPL.setindex!!(vi, x, tm.spl)
    vi  = last(
        DPPL.evaluate_threadunsafe!!(         # we copy vi when doing stuff in parallel so it's ok
            tm.model, vi, DPPL.PriorContext()
        )
    )
    pot = -DPPL.getlogp(vi)
    return pot
end

# evaluate target potential
function V(tm::TuringTemperedModel, x)
    vi  = tm.viout                                 # this helps with the re-binding+Boxing issue: https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
    vi  = DPPL.setindex!!(vi, x, tm.spl)
    DPPL.invlink!(vi, tm.spl)                      # this is to avoid getting the logabsdetjac, which is already in the Vref
    vi  = last(
        DPPL.evaluate_threadunsafe!!(              # we copy vi when doing stuff in parallel so it's ok
            tm.model, vi, DPPL.LikelihoodContext()
        )
    )
    pot = -DPPL.getlogp(vi)
    DPPL.link!(vi, tm.spl)                         # undo the invlink
    return pot
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
# vi = last(DPPL.evaluate!!(lnmodel, vi, mbctx))
# getlogp(vi) ≈ logpdf(LogNormal(),s) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

# link!(vi, spl)
# istrans(vi,@varname(s))
# vi = last(DPPL.evaluate!!(lnmodel, rng, vi, spl, mbctx)) # since mbctx uses DefaultContext, this does not sample, only evaluates
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
# t, state = DPPL.AbstractMCMC.step(rng, lnmodel, spl)
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

# vns = DPPL._getvns(ns.np.V.viout,ns.np.V.spl)
# typeof(vns)
# typeof(vns[1])
# dist = DPPL.getdist(ns.np.V.viout, vns[1][1])
# DPPL.Bijectors.invlink(
#     dist,
#     reconstruct(
#         dist,
#         DPPL.getval(ns.np.V.viout, vns[1])
#     )
# )

#########################################################################################
# Outline of how sampling using Turing.HMC works
#########################################################################################
#
# 1) the sampling starts with a call to "sample"
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L103
#    - after setting default params, this call is redirected to the "mcmcsample" method
#	https://github.com/TuringLang/AbstractMCMC.jl/blob/3de7393b8b8e76330f53505b27d2b928ef178681/src/sample.jl#L90
# 2) the 1st "step" call is done without a "state" object
#	https://github.com/TuringLang/AbstractMCMC.jl/blob/3de7393b8b8e76330f53505b27d2b928ef178681/src/sample.jl#L120
#    For HMC, this call without state is captured here
# 	https://github.com/TuringLang/DynamicPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L71
#    - if no initial parameters are passed, it uses the "initialsampler" method of the sampler to 
#      draw an initial value. Turing defaults to "SampleFromUniform", as shown here
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L100
#    - NOTE: currently "SampleFromUniform" is replaced in DynamicPPL by "SamplerFromPrior". see
# 	https://github.com/TuringLang/DynamicPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L97
#      These samples from there prior occur in the constrained (untransformed) space. 
#    - after setting initial value in constrained space, the "initialstep" method is called
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L143
#      essentially, the function 
#         a) transforms the initial value and compute the logposterior
#         b) creates a Hamiltonian object, which has a metric, a fn to eval the logdens, and a fn
#	     to eval its gradient
#	  c) does a transtion 
#	  d) does adaptation
#	  e) creates a new HMCState, used for the next call 
#	  f) ... 
# 3)   After "initialstep" returns, "mcmcsampling" proceeds using the regular "step!" function, possibly adapting.
#      At each step an "HMCTransition" is created, which holds the parameters in constrained space
#   https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L30   
#      This is achieved by calling "tonamedtuple(vi)", which essentially does vi[@varname(v)] for each variable v.
#   https://github.com/TuringLang/DynamicPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L1022
#      This indexing is the part that implicitly carries out the invlink'ing for a linked vi 
#   https://github.com/TuringLang/DynamicPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L911
#
#      Eventually, sampling finishes, at which point "bundle_samples" is called. this is specialized in Turing,
#      to construct their Chains object from the "samples", which for HMC is a vector of HMCTransitions. Therefore,
#      no need to work with invlink at this stage since the HMCTransitions already have the constrained values.
