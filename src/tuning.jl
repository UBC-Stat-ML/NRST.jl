###############################################################################
# tuning routines
###############################################################################

#######################################
# initialization methods
#######################################

# tune the explorers' parameters
function tune_explorers!(ns::NRSTSampler;nsteps::Int,verbose=false)
    tune!(ns.explorers[1], nsteps = nsteps, verbose=verbose)
    for i in 2:ns.np.N
        # use previous explorer's params as warm start
        pars = params(ns.explorers[i-1])
        tune!(ns.explorers[i], pars, nsteps = nsteps, verbose=verbose)
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

function initialize!(ns::NRSTSampler;nsteps::Int,verbose::Bool=false)
    tune_explorers!(ns;nsteps,verbose)
    initialize_c!(ns;nsteps=2nsteps)
end

#######################################
# tuning using proper runs of the NRST sampler
#######################################

# full tuning
function tune!(
    nss::Vector{<:NRSTSampler};
    init_ntours_per_thread::Int = 32,
    max_rounds::Int = 6,
    max_chng_thrsh::AbstractFloat = 0.025,
    nsteps_expls::Int = max(500, 10*nss[1].nexpl),
    verbose::Bool = true
    )
    ns       = nss[1]
    ntours   = min(128, init_ntours_per_thread * length(nss))
    round    = 0
    max_chng = 1.
    println("\nTuning an NRST sampler using exponentially longer runs.")
    while (max_chng > max_chng_thrsh) && (round < max_rounds)
        round += 1
        verbose && print("Round $round: running $ntours tours...")
        res      = run!(nss, ntours = ntours)           # do a parallel run of ntours
        verbose && print("done!")

        # tune betas
        oldbetas = copy(ns.np.betas)
        tune_betas!(ns, res)
        max_chng = maximum(abs, ns.np.betas - oldbetas) # maximum change in the grid
        verbose && @printf(" Tuned betas, max change = %.2f.", max_chng)

        # adjust explorers' parameters, recompute c, and double tours
        tune_explorers!(ns, nsteps = nsteps_expls, verbose = false)
        initialize_c!(ns, nsteps = min(4000, (2^round)*nsteps_expls))
        verbose && println(" Adjusted explorers and c values.")
        ntours *= 2
    end
end

# tune betas using the equirejection approach
function tune_betas!(ns::NRSTSampler,res::RunResults)
    # estimate Λ at current betas using rejections rates, normalize, and interpolate
    betas = ns.np.betas
    Λnorm, Λvalsnorm = get_lambda(res, betas) # note: this the only place where res is used

    # find newbetas by inverting Λnorm with a uniform grid on the range
    Δ           = 1/ns.np.N      # step size of the grid
    targetΛ     = 0.
    newbetas    = similar(betas)
    newbetas[1] = minimum(betas) # technically 0., but is safe against rounding errors
    for i in 2:ns.np.N
        targetΛ    += Δ
        b1          = newbetas[i-1]
        b2          = betas[findfirst(u -> (u>targetΛ), Λvalsnorm)]            # Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i] = find_zero(β -> Λnorm(β)-targetΛ, (b1,b2), atol = 0.01*Δ) # set tolerance for |Λnorm(β)-target| 
    end
    newbetas[end] = 1.
    copyto!(betas, newbetas) # update betas
end

# estimate Λ at current betas using rejections rates, normalize, and interpolate
function get_lambda(res::RunResults, betas::Vector{<:AbstractFloat})
    rej_rates = res.rejecs ./ res.visits
    averej    = 0.5(rej_rates[1:(end-1),1] + rej_rates[2:end,2]) # average outgoing and incoming rejections
    Λvals     = pushfirst!(cumsum(averej), 0.)
    Λvalsnorm = Λvals/Λvals[end]
    Λnorm     = interpolate(betas, Λvalsnorm, SteffenMonotonicInterpolation())
    @assert sum(abs, Λnorm.(betas) - Λvalsnorm) < 10eps()
    return (Λnorm, Λvalsnorm)
end

# utility for creating the Λ plot
function plot_lambda(Λ,bs,lab)
    c1 = DEF_PAL[1]
    c2 = DEF_PAL[2]
    p = plot(
        x->Λ(x), 0., 1., label = "Λ", legend = :bottomright,
        xlim=(0.,1.), ylim=(0.,1.), color = c1, grid = false
    )
    plot!(p, [0.,0.], [0.,0.], label=lab, color = c2)
    for (i,b) in enumerate(bs[2:end])
        y = Λ(b)
        plot!(p, [b,b], [0.,y], label="", color = c2)                  # vertical segments
        plot!(p, [0,b], [y,y], label="", color = c1, linestyle = :dot) # horizontal segments
    end
    p
end

