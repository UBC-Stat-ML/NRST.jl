###############################################################################
# Using Turing's samplers as ExplorationKernels
# Note: this requires that we run everything in the transformed space, and only
# use constrained space for estimation
# TODO: 
# - NRSTSampler constructor that takes a Turing Model object
#    - will probably need to add a "metadata" field to NRSTProblem to hold the model   
###############################################################################

# # TODO: NRSTSampler constructor
# function NRSTSampler(model::Model, alg::InferenceAlgorithm, betas, nexpl, use_mean)
#     randref = gen_randref(model, spl)
#     x = randref()
#     np = NRSTProblem(V, Vref, randref, betas, similar(betas), use_mean, nothing)
#     explorers = init_explorers(V, Vref, randref, betas, x)
#     # tune explorations kernels and get initial c estimate 
#     initial_tuning!(explorers, np, 10*nexpl)
#     NRSTSampler(np, explorers, x, MVector(0,1), Ref(V(x)), nexpl)
# end


#######################################
# utilities
#######################################

# generate a closure to get a (transformed!) iid sample from the prior
# TODO: it might be more efficient to create (and link!) vi once outside the inner fn,
#       so that it is captured in the closure, and then just use model()
#       i.e., basically recreate the inside of the VarInfo method we use now: https://github.com/TuringLang/DynamicPPL.jl/blob/d222316a7a2fd5afe6ec74a7ec2a50c6f08c1d00/src/varinfo.jl#L120
#       Alternatively, use the rand() approach (it might not work with link! tho): https://github.com/TuringLang/DynamicPPL.jl/blob/d222316a7a2fd5afe6ec74a7ec2a50c6f08c1d00/src/model.jl#L524
function gen_randref(model::Model, spl::Sampler)
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

# generate a function that computes the prior (i.e., reference) potential 
# simplified and modified version of gen_logπ in Turing
# https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L444
# difference: it does not retain the original value in vi, which we don't really need 
function gen_Vref(vi, spl, model)
    function Vref(x)::Float64
        vi  = DynamicPPL.setindex!!(vi, x, spl)
        vi  = last(DynamicPPL.evaluate!!(model, vi, spl, PriorContext()))
        pot = -getlogp(vi)
        return pot
    end
    return Vref
end
