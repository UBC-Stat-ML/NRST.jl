###############################################################################
# tuning routines
# NOTE: as maxcor → 0, the tours lengths become more 
# consistent, so the NRST-par curve in the ESS/cost plot converges to the
# one for MC-par!
###############################################################################

# full tuning
function tune!(
    ns::NRSTSampler{T,I,K};
    max_s1_rounds::Int = 19,
    max_ar_ratio::Real = 0.060,        # limit on std(ar)/mean(ar), ar: average of up/down rejection prob
    max_Δβs::Real      = ns.np.N^(-2), # limit on max change in grid
    max_relΔcone::Real = 0.0015,       # limit on rel change in c(1)
    max_relΔΛ::Real    = 0.015,        # limit on rel change in Λ = Λ(1)
    nsteps_init::Int   = 32,
    max_nsteps::Int    = 8_388_608,
    maxcor::Real       = 0.8,
    min_ntours::Int    = 2_048,
    verbose::Bool      = true
    ) where {T,I,K}
    verbose && println("Tuning started ($(Threads.nthreads()) threads).\n")

    #################################################################
    # stage I: bootstrapping using independent runs from the explorers
    #################################################################

    verbose && println(
        "Stage I: Bootstrapping the sampler using independent runs from the explorers.\n"
    ); show(plot_grid(ns.np.betas, title="Histogram of βs: initial grid")); println("\n")
    rnd     = 0
    nsteps  = nsteps_init÷2                 # nsteps is doubled at the beginning of the loop
    oldcone = relΔcone = oldΛ = relΔΛ = NaN
    conv    = false
    while !conv && (rnd < max_s1_rounds)
        rnd += 1
        nsteps = min(max_nsteps,2nsteps)
        verbose && print("Round $rnd:\n\tTuning explorers...")
        tune_explorers!(ns, verbose = false)
        verbose && println("done!")
        
        # tune c and betas
        verbose && print("\tTuning c and grid using $nsteps steps per explorer...")
        res      = @timed tune_c_betas!(ns, nsteps)
        Δβs, Λ, R= res.value        
        ar       = (R[1:(end-1),1] + R[2:end,2])/2 # note: these rejections are before grid adjustment, so they are technically stale, but are still useful to assess convergence. compute std dev of average of up and down rejs
        ar_ratio = std(ar)/mean(ar)
        relΔcone = abs(ns.np.c[end] - oldcone) / abs(oldcone)
        oldcone  = ns.np.c[end]
        relΔΛ    = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ     = Λ
        verbose && @printf( # the following line cannot be cut with concat ("*") because @printf only accepts string literals
            "done!\n\t\tAR std/mean=%.3f\n\t\tmax(Δbetas)=%.3f\n\t\tc(1)=%.2f (relΔ=%.2f%%)\n\t\tΛ=%.2f (relΔ=%.1f%%)\n\t\tElapsed: %.1fs\n\n", 
            ar_ratio, Δβs, ns.np.c[end], 100*relΔcone, Λ, 100*relΔΛ, res.time
        ); show(plot_grid(ns.np.betas, title="Histogram of βs: round $rnd")); println("\n")

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
    tune_explorers!(ns, verbose = false)
    verbose && print("done!\n\tTuning c and nexpls using $nsteps steps per explorer...")
    res     = @timed @suppress_err tune_c!(ns, nsteps)
    trVs, _ = res.value 
    tune_nexpls!(ns.np.nexpls, trVs, maxcor)
    verbose && @printf("done!\n\t\tElapsed: %.1fs\n\n", res.time)
    verbose && show(lineplot_term(
        ns.np.betas[2:end], ns.np.nexpls, xlabel = "β",
        title="Exploration steps needed to get correlation ≤ $maxcor"
    )); println("\n")
    # after these steps, NRST is coherently tuned and can be used to sample

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
    #      => ntours=nsteps/(2nexpls) gives same comp cost
    #    - but in each of those runs, a explorer returns only 1 sample
    #    - if we split the long runs above into tours, we would get 
    #      nsteps/(2ntours) = nexpls samples per tour per explorer.
    #      => to get same accuracy, need to multiply ntours by sqrt(nexpls)
    # 2) heuristic for when nexpls→∞, where 1) fails
    ntours_f = max(0.5*nsteps/sqrt(mean(ns.np.nexpls)), 150*nsteps^(1/4))
    ntours   = max(min_ntours, min(2nsteps, ceil(Int, ntours_f)))
    verbose && println(
        "\nStage II: tune c(β) using $ntours NRST tours.\n"
    )
    tune_c!(ns, parallel_run(ns, ntours = ntours, keep_xs = false))
    println("\nTuning completed.\n")
    return (nsteps=nsteps, ntours=ntours)
end


##############################################################################
# under the hood
##############################################################################

#######################################
# tune explorers
#######################################

# tune the explorers' parameters
function tune_explorers!(ns::NRSTSampler;smooth=true,kwargs...)
    Threads.@threads for expl in ns.explorers
        tune!(expl;kwargs...)
    end
    smooth && smooth_params!(ns.explorers)
end

# tune the explorers' parameters serially. can use previous expl's params as
# warm start for the next.
function tune_explorers_serial!(ns::NRSTSampler;smooth=true,kwargs...)
    tune!(ns.explorers[1];kwargs...)
    for i in 2:ns.np.N
        # use previous explorer's params as warm start
        pars = params(ns.explorers[i-1])
        tune!(ns.explorers[i], pars;kwargs...)
    end
    smooth && smooth_params!(ns.explorers)
end

#######################################
# utilities for jointly tuning c and grid
#######################################

# Tune c and betas using independent runs of the explorers
# note: explorers must be tuned already before running this
# note: need to tune c first because this is used for estimating rejections (R)
function tune_c_betas!(ns::NRSTSampler, nsteps::Int)
    @unpack c, betas = ns.np
    trVs, _  = @suppress_err tune_c!(ns, nsteps) # tune c using aggV and trVs if necessary
    R        = est_rej_probs(trVs, betas, c)     # compute average rejection probabilities
    (tune_betas!(ns, R)..., R)
end

#######################################
# tune c
#######################################

# tune the c vector using samples of the potential function
function tune_c!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack use_mean, c, betas = ns.np
    if use_mean && abs(aggV[1]) > 1e16
        @info "V likely not integrable under the reference; using stepping stone."
        stepping_stone!(c, betas, trVs) # compute log(Z(b)/Z0) and store it in c
        c .*= (-one(K))                 # c(b) = -log(Z(b)/Z0)
    else
        trapez!(c, betas, aggV)         # trapezodial approximation of int_0^beta aggV(b)db
    end
    return
end

# method for collecting samples using just the explorers
function tune_c!(ns::NRSTSampler, nsteps::Int)
    @unpack c, betas = ns.np
    trVs, aggV = collectVs(ns, nsteps)
    tune_c!(ns, trVs, aggV)
    return (trVs = trVs, aggV = aggV)
end

# method for proper runs of NRST
function tune_c!(ns::NRSTSampler, res::RunResults)
    trVs = res.trVs
    aggV = ns.np.use_mean ? mean.(trVs) : median.(trVs)
    tune_c!(ns, trVs, aggV)
end

#######################################
# tune betas
#######################################

function tune_betas!(ns::NRSTSampler, R::Matrix{<:Real})
    betas    = ns.np.betas
    oldbetas = copy(betas)                       # store old betas to check convergence
    _, _, Λs = optimize_grid!(betas, R)         # tune using the inverse of Λ(β)
    refresh_explorers!(ns)                       # since betas changed, the cached potentials are stale
    return (maximum(abs,betas-oldbetas),Λs[end]) # return max change in grid and Λ=Λ(1) to check convergence
end

# utility to reset Vβ caches in all explorers
function refresh_explorers!(ns::NRSTSampler)
    for e in ns.explorers
        refresh_curVβ!(e)
    end
end

# Tune betas using the output of NRST
# this has the advantage of getting a more accurate representation of the points
# where rejections are higher
tune_betas!(ns::NRSTSampler, res::RunResults) = 
    tune_betas!(ns, res.rpacc ./ res.visits)

# collect samples of V(x) at each of the levels, running explorers independently
# store results in trVs. also compute V aggregate and store in aggV
function collectVs!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack tm, use_mean, N = ns.np
    aggfun = use_mean ? mean : median
    nsteps = length(first(trVs)) 
    for i in 1:nsteps
        trVs[1][i] = V(tm, rand(tm))
    end
    aggV[1] = aggfun(trVs[1])
    Threads.@threads for i in 1:N         # "Threads.@threads for" does not work with enumerate(ns.explorers)
        ipone = i+1
        e = ns.explorers[i]
        run!(e, trVs[ipone])              # run keeping track of V
        aggV[ipone] = aggfun(trVs[ipone])
    end
end

# allocating version
function collectVs(ns::NRSTSampler, nsteps::Int)
    N    = ns.np.N
    aggV = similar(ns.np.c)
    trVs = [similar(aggV, nsteps) for _ in 0:N]
    NRST.collectVs!(ns,trVs,aggV)
    return (trVs = trVs, aggV = aggV)
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
    Threads.@threads for i in 1:(N+1) # "Threads.@threads for" does not work with enumerate(trVs)
        trV = trVs[i]
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
    for i in 1:maxit
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
