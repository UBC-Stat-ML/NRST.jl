###############################################################################
# tuning routines
###############################################################################

# full tuning
function tune!(
    ns::NRSTSampler,
    rng::AbstractRNG;
    min_ntours::Int  = 2_048,
    do_stage_2::Bool = true,
    verbose::Bool    = true,
    kwargs...
    )
    xpls   = replicate(ns.xpl, ns.np.betas)                    # create a vector of explorers, fully independent of ns and its own explorer
    nsteps = tune!(ns.np, xpls, rng;verbose=verbose,kwargs...) # stage I: bootstrap tuning using only independent runs of explorers

    #################################################################
    # Stage II: final c tuning using parallel runs of NRST tours
    # Note: this is desirable particularly in multimodal settings, where independent
    # exploration can be stuck in one mode. In contrast, true NRST sampling is not bound
    # to get stuck in single modes due to the renewal property.
    #################################################################
    
    # two heuristics, choose max of both:
    # 1) rationale:
    #    - above, each explorer produces nsteps samples
    #    - here, every tour each explorer runs twice on average
    #      => ntours=nsteps/(2mean(nexpls)) gives same comp cost on avg
    #    - double it to be safe
    # 2) heuristic for when nexpls→∞, where 1) fails
    ntours_f = max(nsteps/mean(ns.np.nexpls), 150*nsteps^(1/4))
    ntours   = max(min_ntours, min(2nsteps, ceil(Int, ntours_f)))
    if do_stage_2
        verbose && println(
            "\nStage II: tune c(β) using $ntours NRST tours.\n"
        )
        tune_c!(ns.np, parallel_run(ns, rng; ntours = ntours, keep_xs = false))
    end
    println("\nTuning completed.\n")
    return (nsteps=nsteps, ntours=ntours)
end

# stage I tuning: bootstrap independent runs of explorers
function tune!(
    np::NRSTProblem{T,K},
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG;
    max_s1_rounds::Int = 19,
    max_ar_ratio::Real = 0.075,               # limit on std(ar)/mean(ar), ar: average of up/down rejection prob
    max_Δβs::Real      = max(.01, np.N^(-2)), # limit on max change in grid
    max_relΔcone::Real = 0.003,               # limit on rel change in c(1)
    max_relΔΛ::Real    = 0.02,                # limit on rel change in Λ = Λ(1)
    nsteps_init::Int   = 32,
    max_nsteps::Int    = 8_388_608,
    maxcor::Real       = 0.8,
    verbose::Bool      = true
    ) where {T,K}
    verbose && println("Tuning started ($(Threads.nthreads()) threads).\n")

    verbose && println(
        "Stage I: Bootstrapping the sampler using independent runs from the explorers.\n"
    ); show(plot_grid(np.betas, title="Histogram of βs: initial grid")); println("\n")
    rnd     = 0
    nsteps  = nsteps_init÷2                  # nsteps is doubled at the beginning of the loop
    oldcone = relΔcone = oldΛ = relΔΛ = NaN
    conv    = false
    while !conv && (rnd < max_s1_rounds)
        rnd += 1
        nsteps = min(max_nsteps,2nsteps)
        verbose && print("Round $rnd:\n\tTuning explorers...")
        tune_explorers!(xpls, rng, verbose = false)
        verbose && println("done!")
        
        # tune c and betas
        verbose && print("\tTuning c and grid using $nsteps steps per explorer...")
        res      = @timed tune_c_betas!(np, xpls, rng, nsteps)
        Δβs, Λ, R= res.value        
        ar       = (R[1:(end-1),1] + R[2:end,2])/2 # note: these rejections are before grid adjustment, so they are technically stale, but are still useful to assess convergence. compute std dev of average of up and down rejs
        ar_ratio = std(ar)/mean(ar)
        relΔcone = abs(np.c[end] - oldcone) / abs(oldcone)
        oldcone  = np.c[end]
        relΔΛ    = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ     = Λ
        verbose && @printf( # the following line cannot be cut with concat ("*") because @printf only accepts string literals
            "done!\n\t\tAR std/mean=%.3f\n\t\tmax(Δbetas)=%.3f\n\t\tc(1)=%.2f (relΔ=%.2f%%)\n\t\tΛ=%.2f (relΔ=%.1f%%)\n\t\tElapsed: %.1fs\n\n", 
            ar_ratio, Δβs, np.c[end], 100*relΔcone, Λ, 100*relΔΛ, res.time
        ); show(plot_grid(np.betas, title="Histogram of βs: round $rnd")); println("\n")

        # check convergence
        conv = !isnan(relΔcone) && (relΔcone<max_relΔcone) && 
            !isnan(relΔΛ) && (relΔΛ < max_relΔΛ) && (Δβs<max_Δβs) &&
            (ar_ratio < max_ar_ratio)
    end
    # at this point, ns has a fresh new grid, so explorers params, c, and  
    # nexpls are stale => we need to tune them
    verbose && print(
        (conv ? "Grid converged!" :
                "max_rounds=$max_s1_rounds reached.") *
        "\n\nFinal round of stage I: adjusting settings to new grid...\n" *
        "\tTuning explorers..."
    )
    tune_explorers!(xpls, rng, verbose = false)
    for (i,xpl) in enumerate(xpls)         # need to store tuned params for later use with ns.xpl
        np.xplpars[i] = params(xpl)
    end
    verbose && print("done!\n\tTuning c and nexpls using $nsteps steps per explorer...")
    res  = @timed tune_c!(np, xpls, rng, nsteps)
    trVs = res.value
    tune_nexpls!(np.nexpls, trVs, maxcor)
    verbose && @printf("done!\n\t\tElapsed: %.1fs\n\n", res.time)
    verbose && show(lineplot_term(
        np.betas[2:end], np.nexpls, xlabel = "β",
        title="Exploration steps needed to get correlation ≤ $maxcor"
    )); println("\n")
    # after these steps, NRST is coherently tuned and can be used to sample

    return nsteps
end


##############################################################################
# under the hood
##############################################################################

#######################################
# tune explorers
#######################################

# tune the explorers' parameters
function tune_explorers!(
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG;
    smooth=true,
    kwargs...
    )
    N    = length(xpls)
    rngs = [split(rng) for _ in 1:N]
    Threads.@threads for i in 1:N
        tune!(xpls[i],rngs[i];kwargs...)
    end
    smooth && smooth_params!(xpls)
end

#######################################
# utilities for jointly tuning c and grid
#######################################

# Tune c and betas using independent runs of the explorers
# note: explorers must be tuned already before running this
# note: need to tune c first because this is used for estimating rejections (R)
function tune_c_betas!(
    np::NRSTProblem,
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG,
    nsteps::Int
    )
    trVs = tune_c!(np, xpls, rng, nsteps)      # collects Vs, tunes c, and returns Vs
    R    = est_rej_probs(trVs, np.betas, np.c) # compute average rejection probabilities
    out  = tune_betas!(np, R)
    (out..., R)
end

#######################################
# tune c
#######################################

# tune the c vector using samples of the potential function
function tune_c!(np::NRSTProblem{T,K}, trVs::Vector{Vector{K}}) where {T,K}
    stepping_stone!(np.c, np.betas, trVs) # compute log(Z(b)/Z0) and store it in c
    np.c .*= (-one(K))                    # c(b) = -log(Z(b)/Z0)
    return
end

# method for collecting samples using explorers
function tune_c!(
    np::NRSTProblem,
    xpls::Vector{<:ExplorationKernel},
    rng::AbstractRNG,
    nsteps::Int
    )
    trVs = collectVs(np, xpls, rng, nsteps)
    tune_c!(np, trVs)
    return trVs
end

# method for proper runs of NRST
tune_c!(np::NRSTProblem, res::RunResults) = tune_c!(np, res.trVs)

#######################################
# tune betas
#######################################

function tune_betas!(np::NRSTProblem, R::Matrix{<:Real})
    betas    = np.betas
    oldbetas = copy(betas)                       # store old betas to check convergence
    _, _, Λs = optimize_grid!(betas, R)          # tune using the inverse of Λ(β)
    return (maximum(abs,betas-oldbetas),Λs[end]) # return max change in grid and Λ=Λ(1) to check convergence
end

# Tune betas using the output of NRST
# this has the advantage of getting a more accurate representation of the points
# where rejections are higher
tune_betas!(ns::NRSTSampler, res::RunResults) = 
    tune_betas!(ns, res.rpacc ./ res.visits)

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
    Threads.@threads for i in 1:N           # "Threads.@threads for" does not work with enumerate
        ipone = i+1
        update_β!(xpls[i], np.betas[ipone]) # needed because most likely the grid was updated 
        run!(xpls[i], rngs[i], trVs[ipone]) # run keeping track of V
    end
    return trVs
end

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
function optimize_grid!(betas::Vector{K}, R::Matrix{K}) where {K<:AbstractFloat}
    # estimate Λ at current betas using rejections rates, normalize, and interpolate
    f_Λnorm, Λsnorm, Λs = get_lambda(betas, R) # note: this the only place where R is used

    # find newbetas by inverting f_Λnorm with a uniform grid on the range
    N           = length(betas)-1
    Δ           = convert(K,1/N)   # step size of the grid
    Λtargets    = zero(K):Δ:one(K)
    newbetas    = similar(betas)
    newbetas[1] = betas[1]         # technically 0., but is safer this way against rounding errors
    for i in 2:N
        targetΛ     = Λtargets[i]
        b1          = newbetas[i-1]
        b2          = betas[findfirst(u -> (u>targetΛ), Λsnorm)] # f_Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i] = monoroot(β -> f_Λnorm(β)-targetΛ, b1, b2)  # set tolerance for |f_Λnorm(β)-target| 
    end
    newbetas[end] = one(K)
    copyto!(betas, newbetas)
    return (f_Λnorm, Λsnorm, Λs)
end

# estimate Λ at current betas using rejections rates, normalize, and interpolate
function get_lambda(betas::Vector{K}, R::Matrix{K}) where {K<:AbstractFloat}
    averej  = (R[1:(end-1),1] + R[2:end,2])/2 # average up and down rejections
    Λs      = pushfirst!(cumsum(averej), zero(K))
    Λsnorm  = Λs/Λs[end]
    f_Λnorm = interpolate(betas, Λsnorm, SteffenMonotonicInterpolation())
    err     = maximum(abs, f_Λnorm.(betas) - Λsnorm)
    err > 10eps(K) && @warn "get_lambda: high interpolation error = " err
    return (f_Λnorm, Λsnorm, Λs)
end

# tune nexpls by imposing a threshold on autocorrelation
# warning: do not use with trVs generated from NRST, since those include refreshments!
function tune_nexpls!(
    nexpls::Vector{TI},
    trVs::Vector{Vector{TF}},
    maxcor::TF;
    smooth::Bool=false
    ) where {TI<:Int, TF<:AbstractFloat}
    acs = [autocor(trV) for trV in trVs[2:end]]  # compute autocorrelations
    L   = log(maxcor)
    for i in eachindex(nexpls)
        ac  = acs[i]
        idx = findfirst(x->isless(x,maxcor), ac) # attempt to find maxcor in acs
        if !isnothing(idx)
            nexpls[i] = idx - one(TI)            # acs starts at lag 0
        else                                     # extrapolate with model ac[n]=exp(ρn)
            l = length(ac)
            X = reshape(collect(0:(l-1)),(l,1))  # build design matrix
            y = log.(ac)
            ρ = (X \ y)[1]                       # solve least-squares: Xρ ≈ y
            nexpls[i] = ceil(TI, L/ρ)            # x = e^{ρn} => log(x) = ρn => n = log(x)/ρ
        end
    end
    if smooth
        # smooth the results
        N          = length(nexpls)
        lognxs     = log.(nexpls)
        spl        = fit(SmoothingSpline, 1/N:1/N:1, lognxs, .0001)
        lognxspred = predict(spl) # fitted vector
        for i in eachindex(nexpls)
            nexpls[i] = ceil(TI, exp(lognxspred[i]))
        end
    end
end

#######################################
# utilities
#######################################

# find root for monotonic univariate functions
function monoroot(
    f, l::F, u::F;
    tol   = eps(F),
    maxit = 30
    ) where {F<:AbstractFloat}
    fl = f(l)
    fu = f(u)
    if sign(fl) == sign(fu)     # f monotone & same sign at both ends => no root in interval
        return NaN
    end
    h = l
    for _ in 1:maxit
        h  = (l+u)/2
        fh = f(h)
        if abs(fh) < tol
            return h
        elseif sign(fl) == sign(fh)
            l  = h
            fl = fh
        else
            u  = h
            fu = fh
        end
    end
    return h
end

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
