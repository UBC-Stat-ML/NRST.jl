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