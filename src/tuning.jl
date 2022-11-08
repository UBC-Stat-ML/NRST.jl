###############################################################################
# tuning routines
###############################################################################

# "optimal" value for N for a given Λ. γ is a multiplicative constant for 
# correcting the formula when asymptotic assumptions don't hold
optimal_N(Λ, γ) = ceil(Int, γ*Λ*(1+sqrt(1 + inv(1+2Λ))))

# full tuning: bootstraps using ensembles of explorers (NRPT or independent)
function tune!(
    ns::NRSTSampler,
    rng::AbstractRNG;
    min_ntours::Int  = 2_048,
    ensemble::String = "NRPT",
    do_stage_2::Bool = true,
    verbose::Bool    = true,
    kwargs...
    )
    if ensemble == "NRPT"
        ens = NRPTSampler(ns)                                 # PT sampler, whose state is fully independent of ns
    else
        ens = replicate(ns.xpl, ns.np.betas)                  # create a vector of explorers, whose states are fully independent of ns and its own explorer
    end
    oldsumnexpl = sum(ns.np.nexpls)
    nsteps, Λ, TE_ELE = tune!(ns.np, ens, rng;verbose=verbose,kwargs...)

    # If using independent samplers, need to run NRST once to improve c,Λ estimates
    if ensemble != "NRPT" && do_stage_2
        # TODO: this is outdated. just delete?
        # two heuristics for determining appropriate number of tours, choose max of both:
        # 1) rationale:
        #    - above, each explorer produces nsteps samples
        #    - here, every tour each explorer runs twice on average
        #      => ntours=nsteps/(2mean(nexpls)) gives same comp cost on avg
        #    - increase by 50% to be safe: 1/2 * 3/2 = 3/4 = .75 factor in front
        # 2) heuristic for when nexpls→∞, where 1) fails
        ntours_f = max(0.75*nsteps/mean(ns.np.nexpls), 150*nsteps^(1/4))
        ntours   = max(min_ntours, min(2nsteps, ceil(Int, ntours_f)))
        verbose && println("\nImproving c(β) and estimating TE using $ntours NRST tours.\n")
        res = parallel_run(ns, rng; ntours = ntours, keep_xs = false)
        tune_c!(ns.np, res)
    else
        # ntours = min_ntours_TE(TE_ELE,0.8,0.1)
        # cannot use estimated TE_ELE because nexpls changed. if they decreased,
        # then TE_ELE is too optimistic. 
        # 1 PT step = oldsumnexpl steps in the exploration kernels
        # 1 Tour    = 2sum(nexpl) steps in the exploration kernels
        # replace sum(nexpls) -> ns.np.N*median(nexpls) to be robust to cases 
        # where only 1 level has all the exploration steps 
        ntours = min(
            DEFAULT_MAX_TOURS,
            ceil(Int, nsteps * oldsumnexpl / max(1, 2*ns.np.N*median(ns.np.nexpls)) )
        )
    end
    # cannot estimate TE with the ensembles, since this is inherently a regenerative property
    verbose && println("\nEstimating Tour Effectiveness (TE) using $ntours NRST tours.\n")
    res = parallel_run(ns, rng; ntours = ntours, keep_xs = false)
    TE  = last(res.toureff)
    @printf("\nTuning completed! Summary:\n\tTE = %.2f\n\tΛ  = %.2f\n",TE, Λ)
    return TE, Λ
end

# stage I tuning: bootstrap independent runs of explorers
function tune!(
    np::NRSTProblem{T,K},
    ens,                            # ensemble of exploration kernels: either Vector{<:ExplorationKernel} (indep sampling) or NRPTSampler
    rng::AbstractRNG;
    max_rounds::Int    = 14,
    max_ar_ratio::Real = 0.03,      # limit on std(ar)/mean(ar), ar: average of Ru and Rd, the directional rejection rates
    max_dr_ratio::Real = 0.03,      # limit on mean(|Ru-Rd|)/mean(ar). Note: this only makes sense for use_mean=true
    max_Δβs::Real      = 0.05,      # limit on max change in grid. Note: this is not a great indicator, so the limit is quite loose. Only helps with potential fake convergence at beginning
    max_relΔcone::Real = 0.0025,    # limit on rel change in c(1)
    max_relΔΛ::Real    = 0.01,      # limit on rel change in Λ = Λ(1)
    nsteps_init::Int   = 32,
    maxcor::Real       = 0.8,
    γ::Real            = 1.0,       # correction for the optimal_N formula
    check_N::Bool      = true,
    verbose::Bool      = true
    ) where {T,K}
    !np.use_mean && (max_dr_ratio = Inf)      # equality of directional rejections only holds for the mean strategy
    if verbose
        println(
            "Tuning started ($(Threads.nthreads()) threads).\n\n" *
            "Bootstrapping NRST using " *
            (isa(ens, NRPTSampler) ? 
            "NRPT.\n" : 
            "independent runs from the explorers.\n")
        )
        show(plot_grid(np.betas, title="Histogram of βs: initial grid"))
        println("\n")
    end
    rnd     = 0
    nsteps  = nsteps_init÷2                 # nsteps is doubled at the beginning of the loop
    oldcone = relΔcone = oldΛ = relΔΛ = NaN
    Rout    = Matrix{K}(undef, np.N+1, 2)   # capture R matrix generated inside the loop
    conv    = false
    while !conv && (rnd < max_rounds)
        rnd    += 1
        nsteps *= 2        
        verbose && print("Round $rnd:\n\tTuning explorers...")
        tune_explorers!(np, ens, rng) # note: this function forces an update to betas in ens, since grid changed in last round
        verbose && println("done!")
        
        # tune c and betas
        verbose && print("\tTuning c and grid using $nsteps steps per explorer...")
        res        = @timed tune_c_betas!(np, ens, rng, nsteps)
        Δβs,Λ,ar,R = res.value # note: rejections are before grid adjustment, so they are technically stale, but are still useful to assess convergence. compute std dev of average of up and down rejs
        copyto!(Rout, R)
        mar        = mean(ar)
        ar_ratio   = std(ar)/mar
        dr_ratio   = mean(abs, R[1:(end-1),1] - R[2:end,2])/mar
        relΔcone   = abs(np.c[end] - oldcone) / abs(oldcone)
        oldcone    = np.c[end]
        relΔΛ      = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ       = Λ
        if verbose 
            @printf( # the following line cannot be cut with concat ("*") because @printf only accepts string literals
                "done!\n\t\tAR std/mean=%.3f\n\t\tmean|Ru-Rd|/mean=%.3f\n\t\tmax(Δbetas)=%.3f\n\t\tc(1)=%.2f (relΔ=%.2f%%)\n\t\tΛ=%.2f (relΔ=%.1f%%)\n\t\tElapsed: %.1fs\n\n", 
                ar_ratio, dr_ratio, Δβs, np.c[end], 100*relΔcone, Λ, 100*relΔΛ, res.time
            )
            show(plot_grid(np.betas, title="Histogram of βs: round $rnd"))
            println("\n")
        end

        # check if N is too low / high
        if rnd == 5 && check_N 
            Nopt = optimal_N(Λ, γ)
            dN   = abs(np.N-Nopt)
            dN > 1 && dN > .1*Nopt && throw(BadNException(np.N,Nopt))
        end

        # check convergence
        conv = !isnan(relΔcone) && (relΔcone<max_relΔcone) && 
            !isnan(relΔΛ) && (relΔΛ < max_relΔΛ) && (Δβs<max_Δβs) &&
            (ar_ratio < max_ar_ratio) && (dr_ratio < max_dr_ratio)
    end

    # at this point, ns has a fresh new grid, so explorers params, c, and  
    # nexpls are stale => we need to tune them
    verbose && print(
        (conv ? "Grid converged!" :
                "max_rounds=$max_rounds reached.") *
        "\n\nAdjusting settings to new grid...\n" *
        "\tTuning explorers..."
    )
    tune_explorers!(np, ens, rng)
    verbose && print("done!\n\tTuning c and nexpls using $nsteps steps...")
    res = @timed tune_c_nexpls!(np, ens, rng, nsteps, maxcor)
    verbose && @printf("done!\n\t\tElapsed: %.1fs\n\n", res.time)
    verbose && show(lineplot_term(
        np.betas[2:end], np.nexpls, xlabel = "β",
        title="Exploration steps needed to get correlation ≤ $maxcor"
    )); println("\n")

    # after these steps, NRST is coherently tuned and can be used to sample
    return nsteps, oldΛ, toureffELE(Rout)
end


##############################################################################
# under the hood
##############################################################################

##############################################################################
# tune explorers' parameters
##############################################################################

# method for ensemble of independent explorers
function tune_explorers!(
    np::NRSTProblem,
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG;
    smooth=false, # doesn't work well in practice 
    kwargs...
    )
    N    = length(xpls)
    rngs = [split(rng) for _ in 1:N]
    Threads.@threads for i in 1:N
        update_β!(xpls[i], np.betas[i+1]) # needed because most likely the grid was updated 
        tune!(xpls[i], rngs[i];kwargs...)
    end
    smooth && smooth_params!(xpls)
    store_params!(np, xpls)               # need to store tuned params in np for later use with ns.xpl
end

# method for NRPTSampler
function tune_explorers!(np::NRSTProblem, nrpt::NRPTSampler, args...;kwargs...)
    tune_explorers!(np, get_xpls(nrpt), args...;kwargs...)
end

# transfer params from xpls to np for later use
# note: it assumes that xpls are sorted according to levels
function store_params!(np::NRSTProblem, xpls::Vector{<:ExplorationKernel})
    for (i,xpl) in enumerate(xpls)
        np.xplpars[i] = params(xpl)
    end
end

##############################################################################
# utilities for jointly tuning c and grid, exploiting computational savings
##############################################################################

# Tune c and betas using an ensemble of explorers (independent or NRPT)
# note: explorers should be tuned before running this
# note the order: first tune c, then grid. This because c is used for 
# estimating NRST rejections
tune_c_betas!(np::NRSTProblem, ens, args...) = tune_betas!(np, tune_c!(np, ens, args...))

##############################################################################
# tune c
##############################################################################

# tune the c vector using samples of the potential function
function tune_c!(np::NRSTProblem{T,K}, trVs::Vector{Vector{K}}) where {T,K}
    if np.use_mean                             # use mean strategy => can use thermo. ident. to use stepping stone, which is simulation consistent
        stepping_stone!(np.c, np.betas, trVs)  # compute log(Z(b)/Z0) and store it in c
        np.c .*= (-one(K))                     # c(b) = -log(Z(b)/Z0)
    else
        trapez!(np.c, np.betas, median.(trVs)) # use trapezoidal approx of int_0^beta med(b)db
    end
    return
end

# method for collecting samples using ensembles of explorers (indep or NRPT)
function tune_c!(
    np::NRSTProblem, 
    ens::Union{NRPTSampler, Vector{<:ExplorationKernel}},
    args...
    )
    trVs = collectVs(np, ens, args...)
    tune_c!(np, trVs)
    return trVs
end

# method for proper runs of NRST, which contain samples of V
tune_c!(np::NRSTProblem, res::RunResults) = tune_c!(np, res.trVs)

##############################################################################
# tune betas
##############################################################################

# uses the average of the up and down rates of rejection
function tune_betas!(np::NRSTProblem, ar::Vector{<:Real})
    betas    = np.betas
    oldbetas = copy(betas)                       # store old betas to check convergence
    _, _, Λs = optimize_grid!(betas, ar)         # tune using the inverse of Λ(β)
    return (maximum(abs,betas-oldbetas),Λs[end]) # return max change in grid and Λ=Λ(1) to check convergence
end

# method for the raw traces of V along the levels
function tune_betas!(np::NRSTProblem{T,K}, trVs::Vector{Vector{K}}) where {T,K}
    R   = est_rej_probs(trVs, np.betas, np.c) # compute average directional NRST rejection probabilities
    ar  = averej(R)                           # average up and down directions
    out = tune_betas!(np, ar)
    (out..., ar, R)
end

# method for the output of NRST
tune_betas!(np::NRSTProblem, res::RunResults) = tune_betas!(np, averej(res))

# collect samples of V(x) at each of the levels, running explorers independently
# also compute V aggregate
# note: np.tm is modified here because it is used for sampling V from the reference
function collectVs(
    np::NRSTProblem, 
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG,
    nsteps::Int
    )
    N      = np.N
    trVs   = [similar(np.c, nsteps) for _ in 0:N]
    for i in 1:nsteps
        trVs[1][i] = V(np.tm, rand(np.tm, rng))
    end
    rngs = [split(rng) for _ in 1:N]
    Threads.@threads for i in 1:N          # "Threads.@threads for" does not work with enumerate
        xpl = xpls[i]
        rng = rngs[i]
        update_β!(xpl, np.betas[i+1])      # needed because most likely the grid was updated
        ar  = run!(xpl, rng, trVs[i+1])    # run and collect Vs
        while ar < .05                     # if acc too low, re-tune and retry
            @debug "Re-tuning explorer $i due to low acc-rate = $round(ar,digits=2)."
            tune!(xpl, rng)
            ar = run!(xpl, rng, trVs[i+1])
        end
    end
    return trVs
end

# method for NRPTSampler
collectVs(::NRSTProblem, nrpt::NRPTSampler, args...) = rows2vov(run!(nrpt, args...).Vs)

# for each sample V(x) at each level, estimate the conditional rejection
# probabilities in both directions. Recall that
#   ap(x) = min{1,exp(-[(β' - β)*V(x) - (c' - c)])}
# setting
#   ΔE(x) := (β' - β)*V(x) - (c' - c)
# we get
#   ap(x) = min{1,exp(-ΔE(x))} = exp(log(min{1,exp(-ΔE(x))})) 
#         = exp(min{0,-ΔE(x)})) = exp(-max{0,ΔE(x)}))
# we can only move up or down, so
#   ΔEup_i(x) := (β[i+1] - β[i])*V(x) - (c[i+1] - c[i]), i in 0:(N-1)
#   ΔEdn_i(x) := (β[i-1] - β[i])*V(x) - (c[i-1] - c[i]), i in 1:N
# setting 
#   dupβ[i] := β[i+1] - β[i]
# we have
#   ddnβ[i] := β[i-1] - β[i] = -(β[i]-β[i-1]) = -dupβ[i-1]
# Finally, we use the expm1 identity for increased numerical stability
#   1 - exp(u) = -(exp(u) - 1) = -expm1(u)
function est_rej_probs(trVs, betas, c)
    N    = length(c)-1
    dbs  = diff(betas)          # betas[i+1] - betas[i] for i ∈ 1:(length(betas)-1)
    dcs  = diff(c)              # c[i+1] - c[i] for i=1:(length(c)-1)
    R    = similar(c, (N+1, 2)) # rejection probs: R[:,1] is up, R[:,2] is dn
    uno  = one(eltype(c))
    cero = zero(eltype(c))
    for (i, trV) in enumerate(trVs)
        if i > N
            R[i,1] = uno
        else
            R[i,1] = -mean([expm1(-max(cero, dbs[i]*v-dcs[i])) for v in trV])
        end
        if i <= 1
            R[i,2] = uno
        else
            R[i,2] = -mean([expm1(-max(cero, -dbs[i-1]*v+dcs[i-1])) for v in trV])
        end
    end
    return R
end

# optimize the grid using the equirejection approach
# returns estimates of Λ(β) at the grid locations
function optimize_grid!(betas::Vector{K}, averej::Vector{K}) where {K<:AbstractFloat}
    # estimate Λ at current betas using rejections rates, normalize, and interpolate
    f_Λnorm, Λsnorm, Λs = gen_lambda_fun(betas, averej) # note: this the only place where averej is used

    # find newbetas by inverting f_Λnorm with a uniform grid on the range
    N           = length(betas)-1
    Δ           = convert(K,1/N)   # step size of the grid
    Λtargets    = zero(K):Δ:one(K)
    newbetas    = similar(betas)
    newbetas[1] = betas[1]         # technically 0., but is safer this way against rounding errors
    for i in 2:N
        targetΛ       = Λtargets[i]
        b1            = newbetas[i-1]
        b2            = betas[findfirst(u -> (u>targetΛ), Λsnorm)] # f_Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i],_ = monoroot(β -> f_Λnorm(β)-targetΛ, b1, b2)
    end
    newbetas[end] = one(K)
    copyto!(betas, newbetas)
    return (f_Λnorm, Λsnorm, Λs)
end

# estimate Λ, normalize, and interpolate
function gen_lambda_fun(betas::Vector{K}, averej::Vector{K}) where {K<:AbstractFloat}
    Λs      = get_lambdas(averej)
    Λsnorm  = Λs/Λs[end]
    f_Λnorm = interpolate(betas, Λsnorm, FritschButlandMonotonicInterpolation()) # more natural fit than SteffenMonotonicInterpolation
    err     = maximum(abs, map(f_Λnorm, betas) - Λsnorm)
    err > 10eps(K) && @warn "gen_lambda_fun: high interpolation error = " err
    return (f_Λnorm, Λsnorm, Λs)
end

# estimate Λs at current betas using rejections rates
function get_lambdas(averej::Vector{K}) where {K<:AbstractFloat}
    pushfirst!(cumsum(averej), zero(K))
end

##############################################################################
# tune nexpls by imposing a threshold on autocorrelation
# warning: do not use with trVs generated from NRST, since those include refreshments!
##############################################################################

function tune_nexpls!(
    nexpls::Vector{TI},
    trVs::Vector{Vector{TF}},
    maxcor::TF;
    smooth::Bool=false
    ) where {TI<:Int, TF<:AbstractFloat}
    L = log(maxcor)
    for i in eachindex(nexpls)
        trV = any(trVs[i+1] .> 1e17) ? winsor(trVs[i+1], prop=0.01) : trVs[i+1]
        std(trV) < eps(TF) && throw(ArgumentError("Explorer $i produced constant V samples."))
        ac  = autocor(trV)
        idx = findfirst(a -> a<=maxcor, ac)      # attempt to find maxcor in acs
        if !isnothing(idx)
            nexpls[i] = idx - one(TI)            # acs starts at lag 0
        else                                     # extrapolate with model ac[n]=exp(ρn)
            l  = length(ac)
            xs = 0:(l-1)
            ys = log.(ac)
            ρ  = sum(xs .* ys) / sum(abs2, xs)   # solve least-squares: y ≈ Xρ
            try
                nexpls[i] = ceil(TI, L/ρ)        # x = e^{ρn} => log(x) = ρn => n = log(x)/ρ
            catch e
                @warn "tune_nexpls!: caught error when setting nexpl for i=$i. Dumping info:\n"
                println("ac:");display(ac)
                println("First 10 V values:"); display(trVs[i+1][1:10])
                println("Last 10 V values:"); display(trVs[i+1][(end-9):end])
                println("trV contains $(sum(isnan, trVs[i+1])) NaN values")
                rethrow(e)
            end
        end
    end
    if smooth
        # smooth the results. note: not very useful in practice
        N          = length(nexpls)
        lognxs     = log.(nexpls)
        spl        = fit(SmoothingSpline, 1/N:1/N:1, lognxs, .0001)
        lognxspred = predict(spl) # fitted vector
        for i in eachindex(nexpls)
            nexpls[i] = ceil(TI, exp(lognxspred[i]))
        end
    end
end

# jointly tune c and nexpls: makes sense because both need samples form V
# method for ensemble of iid explorers
function tune_c_nexpls!(
    np::NRSTProblem,
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG,
    nsteps::Int,
    maxcor::AbstractFloat
    )
    tune_nexpls!(np.nexpls, tune_c!(np, xpls, rng, nsteps), maxcor)
end

# method for NRPT: use NRPT for tuning c (higher quality), and then extract
# explorers and run the method above for tuning nexpls. This is needed because
# NRPT traces are "contaminated" with swaps.
function tune_c_nexpls!(
    np::NRSTProblem,
    nrpt::NRPTSampler,
    rng::AbstractRNG,
    nsteps::Int,
    maxcor::AbstractFloat
    )
    _    = tune_c!(np, nrpt, rng, nsteps)
    xpls = get_xpls(nrpt)
    trVs = collectVs(np, xpls, rng, np.nexpls[1]*nsteps) # make equivalent effort to an NRPT step
    tune_nexpls!(np.nexpls, trVs, maxcor)
end

##############################################################################
# utilities
##############################################################################

#######################################
# plotting utilities
#######################################

# histogram of the grid using UnicodePlots
function plot_grid(bs;kwargs...)
    len = min(displaysize(stdout)[2]-8, 70)
    UnicodePlots.histogram(
        bs;
        xlabel  = "β",
        stats   = false,
        nbins   = len,
        vertical=true,
        width   = len,
        height  = 1,
        grid    = false,
        border  = :none,
        # yticks  = false,
        margin  = 0,
        padding = 0,
        kwargs...
    )    
end

# generic lineplot using UnicodePlots
function lineplot_term(xs,ys;kwargs...)
    len = min(displaysize(stdout)[2]-8, 70)
    UnicodePlots.lineplot(
        xs, ys;
        width   = len,
        height  = 6,
        grid    = false,
        border  = :none,
        margin  = 0,
        padding = 0,
        kwargs...
    )
end
