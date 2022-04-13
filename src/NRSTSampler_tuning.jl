###############################################################################
# tuning routines
###############################################################################

#######################################
# full tuning methods
#######################################

function tune!(
    explorers::AbstractVector{<:ExplorationKernel},
    np::NRSTProblem;
    nsteps::Int
    )
    @unpack c, betas, fns = np
    @unpack V, randref = fns
    aggfun  = np.use_mean ? mean : median
    aggV    = similar(c)
    aggV[1] = aggfun([V(randref()) for _ in 1:nsteps])
    traceV  = similar(c, nsteps)
    for (i,e) in enumerate(explorers)
        tune!(e, V, traceV)
        aggV[i+1] = aggfun(traceV)
    end
    trapez!(c,betas,aggV) # trapezoidal approx of int_0^beta db aggV(b)
end
tune!(ns::NRSTSampler; kwargs...) = tune!(ns.explorers, ns.np; kwargs...)
tune!(nss::Vector{<:NRSTSampler}; kwargs...) = tune!(nss[1]; kwargs...)

#######################################
# Tune the c params using the output of serial or parallel run
#######################################

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

