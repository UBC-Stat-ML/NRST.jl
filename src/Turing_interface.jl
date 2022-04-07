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
    randref = gen_randref(spl, model)
    V       = gen_V(viout, spl, model)
    Vref    = gen_Vref(viout, spl, model)
    NRSTSampler(V, Vref, randref, betas, nexpl, use_mean)
end

#######################################
# utilities
#######################################

# generate a closure to get a (transformed!) iid sample from the prior
# TODO: it might be more efficient to create (and link!) vi once outside the inner fn,
#       so that it is captured in the closure, and then just use model()
#       i.e., basically recreate the inside of the VarInfo method we use now: https://github.com/TuringLang/DynamicPPL.jl/blob/d222316a7a2fd5afe6ec74a7ec2a50c6f08c1d00/src/varinfo.jl#L120
#       Alternatively, use the rand() approach (it might not work with link! tho): https://github.com/TuringLang/DynamicPPL.jl/blob/d222316a7a2fd5afe6ec74a7ec2a50c6f08c1d00/src/model.jl#L524
function gen_randref(spl, model)
    # TODO: this is not type stable. could be fixed with an approach similar to their rand() method 
    #       Another option is to use the fact that vi is typed, and so can 
    #       use eltype(vi,spl), but this would need creating vi outside the inner fun
    #       Another similar op: construct the typed vi and then get TVal from the metadata: https://github.com/TuringLang/DynamicPPL.jl/blob/d222316a7a2fd5afe6ec74a7ec2a50c6f08c1d00/src/varinfo.jl#L58 
    function randref(rng::AbstractRNG=Random.GLOBAL_RNG)
        vi = DynamicPPL.VarInfo(rng, model, SampleFromPrior(), PriorContext()) # avoids evaluating the likelihood
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
# TODO: this is inefficient because it requires 2 passes over the graph to get Vref + βV
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
        vi  = viout # https://invenia.github.io/blog/2019/10/30/julialang-features-part-1/#an-aside-on-boxing
        vi  = DynamicPPL.setindex!!(vi, x, spl)
        vi  = last(DynamicPPL.evaluate!!(model, vi, spl, LikelihoodContext()))
        pot = -getlogp(vi)
        return pot
    end
    return V
end

