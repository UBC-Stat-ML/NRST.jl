###############################################################################
# Using Turing's HMC samplers as ExplorationKernels
# TODO: 
# - NRSTSampler constructor that takes a Turing Model object
#    - will probably need to add a "metadata" field to NRSTProblem to hold the model   
###############################################################################




#######################################
# utilities
#######################################

# generate a closure to get a (transformed!) iid sample from the prior
function gen_randref(model::Model, spl::Sampler)
    function randref(rng::AbstractRNG=Random.GLOBAL_RNG)
        vi = DynamicPPL.VarInfo(rng, model, SampleFromPrior())
        DynamicPPL.link!(vi, spl)
        vi[spl]
    end
    return randref
end

# # TODO: tempered version gen_temp_logπ
# function gen_logπβ(vi_base, spl::AbstractSampler, model, invtemp)
#     function logπβ(x)::Float64
#         vi = vi_base
#         x_old, lj_old = vi[spl], getlogp(vi)
#         vi = setindex!!(vi, x, spl)
#         vi = last(DynamicPPL.evaluate!!(model, vi, spl))
#         lj = getlogp(vi)
#         # Don't really need to capture these will only be
#         # necessary if `vi` is indeed mutable.
#         setindex!!(vi, x_old, spl)
#         setlogp!!(vi, lj_old)
#         return lj
#     end
#     return logπ
# end