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
    initialize_c!(ns;nsteps=2nsteps)
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
# tune betas using the equirejection approach
#######################################

function tune_betas!(ns::NRSTSampler,res::RunResults; visualize=false)
    # estimate Λ at current betas using the rejections, normalize, and interpolate
    rejrat    = res.rejecs ./ res.visits                   # note: this the only place where res is used
    averej    = 0.5(rejrat[1:(end-1),1] + rejrat[2:end,2]) # average outgoing and incoming rejections
    Λvals     = pushfirst!(cumsum(averej), 0.)
    Λvalsnorm = Λvals/Λvals[end]
    betas     = ns.np.betas
    Λnorm     = interpolate(betas, Λvalsnorm, SteffenMonotonicInterpolation())
    @assert sum(abs, Λnorm.(betas) - Λvalsnorm) < 10eps()

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
    
    if visualize
        p1 = plot_lambda(Λnorm, betas, "βold")
        p2 = plot_lambda(Λnorm, newbetas, "βnew")
        display(plot(p1, p2, layout = (2,1)))
        copyto!(betas, newbetas)
        return (p1,p2)
    else
        copyto!(betas, newbetas)
        return
    end

end

# utility for creating the Λ plot
function plot_lambda(Λ,bs,lab)
    c1 = DEFAULT_PALETTE[1]
    c2 = DEFAULT_PALETTE[2]
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

