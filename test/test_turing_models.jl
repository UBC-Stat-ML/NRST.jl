#########################################################################################
# interfacing with the Turing.jl environment
# TODO:
# - use DynamicPPL @models, evaluate V_b(x) := Vref(x) + bV(x) with MiniBatchContext()
#########################################################################################

using Turing


#########################################################################################
# TODO:
# - use AdvancedHMC samplers as exploration kernels. Cannot really use the Turing because
#   the model for the explorer is only specified through the tempered energy V_b, which is
#   more similar to the standard AdvancedHMC interface. however, must be careful about link/unlik
#   to handle transformations 
#
# Outline of how Turing adapts AdvancedHMC to work with trans variables, within AdvancedMCMC framework
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



#########################################################################################
# OLD
#########################################################################################
# test statistical models defined in Turing.jl using the 2 helper functions
# power_logjoint and get_loglikelihood
# defined in https://github.com/theogf/ThermodynamicIntegration.jl/blob/main/src/turing.jl
# here I'll just write a disection of the code for eventually adapting it and
# hopefully improving it
# NOTE: we also need sampling from the prior!
# also this is perhaps a better way
# https://github.com/TuringLang/DynamicPPL.jl/issues/112#issuecomment-627474180
# helper functions implemented starting here
# https://github.com/TuringLang/DynamicPPL.jl/blob/dd1d30115dea98885d5e03ce361b376d131e1b28/src/model.jl#L497
function power_logjoint(model, β)
    ctx = DynamicPPL.MiniBatchContext(DynamicPPL.DefaultContext(), β)
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        
        # this one is matched here
        # https://github.com/TuringLang/DynamicPPL.jl/blob/ad545beaab3f104d62a01d926ad2d2998cb41614/src/model.jl#L395
        # which then goes tp here
        # https://github.com/TuringLang/DynamicPPL.jl/blob/ad545beaab3f104d62a01d926ad2d2998cb41614/src/model.jl#L377
        model(varinfo, spl, ctx)
        
        return DynamicPPL.getlogp(varinfo)
    end
end

function get_loglikelihood(model)
    ctx = DynamicPPL.LikelihoodContext()
    spl = DynamicPPL.SampleFromPrior()
    vi = DynamicPPL.VarInfo(model)
    return function f(z)
        varinfo = DynamicPPL.VarInfo(vi, spl, z)
        model(varinfo, spl, ctx)
        return DynamicPPL.getlogp(varinfo)
    end
end
