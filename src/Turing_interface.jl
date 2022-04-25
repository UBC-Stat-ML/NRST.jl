###############################################################################
# Minimal interface that lets one obtain randref, Vref and V from a DPPL.Model
# The transformation 𝕏 → ℝ from constrained to unconstrained Euclidean space
# is used to simplify the work for exploration kernels 
###############################################################################

const DPPL = DynamicPPL

# storage for functions associated to a tempered problem
struct TuringFuns{TV,TVr,Tr,Tmod,Tspl,TVi} <: Funs
    V::TV       # energy Function
    Vref::TVr   # energy of reference distribution
    randref::Tr # produces independent sample from reference distribution
    model::Tmod # a DPPL.Model
    spl::Tspl   # a DPPL.Sampler
    viout::TVi  # a DPPL.VarInfo
end

function TuringFuns(model::DPPL.Model)
    viout   = DPPL.VarInfo(model)             # build a TypedVarInfo
    spl     = DPPL.Sampler(Turing.HMC(0.1,5)) # build a dummy sampler that works in unconstrained space
    DPPL.link!(viout, spl)                    # force transformation 𝕏 → ℝ
    randref = gen_randref(viout, spl, model)
    V       = gen_V(viout, spl, model)
    Vref    = gen_Vref(viout, spl, model)
    TuringFuns(V, Vref, randref, model, spl, viout)
end

# copy a TuringFuns. it keeps model and spl common. Avoiding copying model is
# especially important because it contains the dataset, which can be huge
function Base.copy(fns::TuringFuns)
    @unpack model, spl = fns
    vinew   = DPPL.VarInfo(model)            # build a new TypedVarInfo
    DPPL.link!(vinew, spl)                   # link with old sampler to force transformation 𝕏 → ℝ
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
function NRSTSampler(model::DPPL.Model, args...;kwargs...)
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
        vi::Tvi = DPPL.VarInfo(rng, model, DPPL.SampleFromPrior(), DPPL.PriorContext()) # avoids evaluating the likelihood
        DPPL.link!(vi, spl)
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
        vi  = DPPL.setindex!!(vi, x, spl)
        vi  = last(DPPL.evaluate!!(model, vi, spl, DPPL.PriorContext()))
        pot = -DPPL.getlogp(vi)
        return pot
    end
    return Vref
end
function gen_V(viout, spl, model)
    function V(x)::Float64
        vi  = viout # this helps with the re-binding+Boxing issue: https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
        vi  = DPPL.setindex!!(vi, x, spl)
        vi  = last(DPPL.evaluate!!(model, vi, spl, DPPL.LikelihoodContext()))
        pot = -DPPL.getlogp(vi)
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
# 	https://github.com/TuringLang/DPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L71
#    - if no initial parameters are passed, it uses the "initialsampler" method of the sampler to 
#      draw an initial value. Turing defaults to "SampleFromUniform", as shown here
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L100
#    - NOTE: currently "SampleFromUniform" is replaced in DPPL by "SamplerFromPrior". see
# 	https://github.com/TuringLang/DPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L97
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
#   https://github.com/TuringLang/DPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L1022
#      This indexing is the part that implicitly carries out the invlink'ing for a linked vi 
#   https://github.com/TuringLang/DPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L911
#
#      Eventually, sampling finishes, at which point "bundle_samples" is called. this is specialized in Turing,
#      to construct their Chains object from the "samples", which for HMC is a vector of HMCTransitions. Therefore,
#      no need to work with invlink at this stage since the HMCTransitions already have the constrained values.
