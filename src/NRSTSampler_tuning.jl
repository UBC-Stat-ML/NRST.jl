###############################################################################
# tuning routines
###############################################################################

# tune the explorers' parameters
function tune_explorers!(ns::NRSTSampler;nsteps::Int)
    tune!(ns.explorers[1], nsteps = nsteps)
    for i in 2:ns.np.N
        # use previous explorer's params as warm start
        pars = params(ns.explorers[i-1])
        tune!(ns.explorers[i], pars, nsteps = nsteps)
    end
end

# Tune the c params using independent runs of the explorers
# this is a safer way for initially tuning c
function initialize_c!(ns::NRSTSampler;nsteps::Int)
    @unpack c, betas, fns, use_mean = ns.np
    @unpack V, randref = fns
    aggfun  = use_mean ? mean : median
    aggV    = similar(c)
    aggV[1] = aggfun([V(randref()) for _ in 1:nsteps])
    traceV  = similar(c, nsteps)
    for (i,e) in enumerate(ns.explorers)
        run!(e, V, traceV)
        aggV[i+1] = aggfun(traceV)
    end
    trapez!(c,betas,aggV) # trapezoidal approx of int_0^beta db aggV(b)
end

function initialize!(ns::NRSTSampler;nsteps::Int)
    tune_explorers!(ns;nsteps)
    initialize_c!(ns;nsteps)
end

# Tune the c params using the output of serial or parallel run
function tune_c!(ns::NRSTSampler,res::RunResults)
    @unpack np = ns
    @unpack c, N, betas, fns, use_mean = np
    aggfun = use_mean ? mean : median
    aggV   = point_estimate(res, h=fns.V, at=1:(N+1), agg=aggfun)
    trapez!(c, betas, aggV) # use trapezoidal approx to estimate int_0^beta db aggV(b)
end

#######################################
# TODO: tune betas
# for aggfun=mean, use the PT equirejection approach
#######################################

