###############################################################################
# second-stage tuning (i.e., after initial_tuning)
# uses the output of postprocess_tours
###############################################################################

function tune_c!(ns::NRSTSampler,res::ParallelRunResults)
    @unpack np = ns
    @unpack c, betas, V, use_mean = np
    aggfun = use_mean ? mean : median
    aggV = NRST.point_estimate(res, V, aggfun)
    trpz_apprx!(c, betas, aggV) # use trapezoidal approx to estimate int_0^beta db aggV(b)
end

# run in parallel multiple rounds with an exponentially increasing number of tours
# NOTE: the np field is shared across samplers so changing just one affects all
# TODO: this is a REAALLLY bad procedure in practice, for some reason simply doing
#    NRST.initial_tuning!(ns.explorers, ns.np, 2000)
# is so much better! Basically, the latter is more efficient by not dealing with
# the index process: it only works with the explorers!
function tune!(
    samplers::Vector{<:NRSTSampler};
    nrounds::Int  = 5,
    nthrds::Int   = Threads.nthreads(),
    ntours0::Int  = 32,
    verbose::Bool = false
    )
    ntours = ntours0 # initialize number of tours
    for nr in 1:nrounds
        if verbose
            println("Tuning round $nr with $ntours tours per thread")
            println("Current c:")
            show(samplers[1].np.c)
            println("")
        end
        res = parallel_run!(samplers, ntours = ntours*nthrds)
        tune_c!(samplers[1], res)
        ntours *= 2
    end
end
