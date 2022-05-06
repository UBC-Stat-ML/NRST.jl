###############################################################################
# tuning routines
###############################################################################

# full tuning
function tune!(
    ns::NRSTSampler{T,I,K};
    max_s1_rounds::Int = 12,
    max_s2_rounds::Int = 3,
    max_Δβs::Real      = 0.01,
    max_relΔcone::Real = 0.001,
    max_relΔΛ::Real    = 0.005,
    nsteps_init::Int   = 32,
    max_nsteps::Int    = 65_536,
    maxcor::Real       = 0.95,
    ntours::Int        = 6_192,
    verbose::Bool      = true
    ) where {T,I,K}
    verbose && println("Tuning started ($(Threads.nthreads()) threads).\n")

    #################################################################
    # stage I: bootstrapping using independent runs from the explorers
    #################################################################

    verbose && println(
        "Stage I: Bootstrapping the sampler using independent runs from the explorers.\n"
    ); show(plot_grid(ns.np.betas, title="Histogram of βs: initial grid")); println("\n")
    rnd   = 0
    nsteps  = nsteps_init÷2                 # nsteps is doubled at the beginning of the loop
    oldcone = relΔcone = oldΛ = relΔΛ = NaN # to assess convergence on c(1) and Λ=Λ(1)
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
        Δβs, Λ   = res.value
        relΔcone = abs(ns.np.c[end] - oldcone) / abs(oldcone)
        oldcone  = ns.np.c[end]
        relΔΛ    = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ     = Λ
        verbose && @printf(
            "done!\n\t\tmax(Δbetas)=%.3f\n\t\tc(1)=%.2f (relΔ=%.2f%%)\n\t\tΛ=%.2f (relΔ=%.1f%%)\n\t\tElapsed: %.1fs\n\n", 
            Δβs, ns.np.c[end], 100*relΔcone, Λ, 100*relΔΛ, res.time
        ); show(plot_grid(ns.np.betas, title="Histogram of βs: round $rnd")); println("\n")

        # check convergence
        conv = !isnan(relΔcone) && (relΔcone<max_relΔcone) && 
            !isnan(relΔΛ) && (relΔΛ < max_relΔΛ) && (Δβs<max_Δβs)
    end
    
    #################################################################
    # final round of stage I: since betas changed the last, we need 
    # to tune explorers and c one last time. We also set nexpls.
    #################################################################

    verbose && print(
        (conv ? "Grid converged!" :
                "max_rounds=$max_s1_rounds reached.") *
        "\nFinal round of stage I: adjusting settings to new grid...\n" *
        "\tTuning explorers..."
    )
    tune_explorers!(ns, verbose = false)
    verbose && println("done!\n\tTuning c and nexpls using $nsteps steps per explorer...\n")
    trVs, _ = tune_c!(ns, nsteps)
    tune_nexpls!(ns.np.nexpls, trVs, maxcor)
    verbose && show(lineplot_term(
        ns.np.betas[2:end], ns.np.nexpls, xlabel = "β",
        title="Exploration steps needed to get correlation ≤ $maxcor"
    )); println("\n")

    # Note: at this point, ns is adequately (but not optimally) tuned,
    # so we can always just finish here if we are in a hurry?

    #################################################################
    # Stage II: tuning using parallel runs of NRST tours
    # Note: this is desirable particularly in multimodal settings, where independent
    # exploration can be stuck in one mode. In contrast, true NRST sampling is not bound
    # to get stuck in single modes due to the renewal property.
    #################################################################
    
    verbose && println(
        "\nStage II: tuning using NRST tours run in parallel.\n"
    )
    rnd  = 0
    conv = false
    while !conv && (rnd < max_s2_rounds)
        rnd += 1
        
        # tune grid and then c
        itp_c = LinearInterpolation(ns.np.betas, ns.np.c)
        verbose && println(
            "Round $rnd:\n\tTuning c and grid using $ntours tours...\n"
        )
        Δβs, Λ   = tune_c_betas!(ns, parallel_run(ns, ntours = ntours))
        relΔcone = abs(ns.np.c[end] - oldcone) / abs(oldcone)
        oldcone  = ns.np.c[end]
        relΔΛ    = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ     = Λ
        verbose && @printf(
            "\n\tdone!\n\t\tmax(Δbetas)=%.3f\n\t\tc(1)=%.2f (relΔ=%.2f%%)\n\t\tΛ=%.2f (relΔ=%.1f%%)\n", 
            Δβs, ns.np.c[end], 100*relΔcone, Λ, 100*relΔΛ
        ); show(plot_grid(ns.np.betas, title="Histogram of βs: round $rnd")); println("\n")
        verbose && print("\tTuning explorers...")
        tune_explorers!(ns, verbose = false)
        verbose && println("done!\n\tTuning c and nexpls using $nsteps steps per explorer...\n")
        trVs, _ = @suppress_err tune_c!(ns, nsteps)
        tune_nexpls!(ns.np.nexpls, trVs, maxcor)
        verbose && show(lineplot_term(
            ns.np.betas[2:end], ns.np.nexpls, xlabel = "β",
            title="Exploration steps needed to get correlation ≤ $maxcor"
        )); println("\n")

        # check convergence
        conv = !isnan(relΔcone) && (relΔcone<max_relΔcone) && 
            !isnan(relΔΛ) && (relΔΛ < max_relΔΛ) && (Δβs<max_Δβs)
        ntours += 1_000
        nsteps  = min(max_nsteps,2nsteps)
    end

    #################################################################
    # Stage III: final c tuning using parallel runs of NRST tours
    #################################################################
    
    verbose && println(
        "\nStage III: final c tuning using parallel runs of NRST tours.\n"
    )
    tune_c!(ns, parallel_run(ns, ntours = ntours))
    println("\nTuning completed.\n")
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
    tune_betas!(ns, R)
end

# method for proper runs of NRST
# important: must do c first, because after changing betas "res" is stale
function tune_c_betas!(ns::NRSTSampler, res::RunResults)
    tune_c!(ns, res)
    tune_betas!(ns, res)
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
tune_betas!(ns::NRSTSampler, res::RunResults) = tune_betas!(ns, res.rpacc ./ res.visits)

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
        b2          = betas[findfirst(u -> (u>targetΛ), Λsnorm)]                 # f_Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i] = find_zero(β -> f_Λnorm(β)-targetΛ, (b1,b2), atol = Δ*1e-7) # set tolerance for |f_Λnorm(β)-target| 
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
    maxcor::TF
    ) where {TI<:Int, TF<:AbstractFloat}
    acs = [autocor(trV) for trV in trVs[2:end]]  # compute autocorrelations
    L   = log(maxcor)
    for i in eachindex(nexpls)
        ac  = acs[i]
        idx = findfirst(x->isless(x,maxcor), ac) # attempt to find maxcor in acs
        if !isnothing(idx)
            nexpls[i] = idx - 1                  # acs starts at lag 0
        else                                     # extrapolate with model ac[n]=exp(ρn)
            l = length(ac)
            X = reshape(collect(0:(l-1)),(l,1))  # build design matrix
            y = log.(ac)
            ρ = (X \ y)[1]                       # solve least-squares: Xρ ≈ y
            nexpls[i] = ceil(TI, L/ρ)            # x = e^{ρn} => log(x) = ρn => n = log(x)/ρ
        end
    end
    # smooth the results
    N          = length(nexpls)
    lognxs     = log.(nexpls)
    spl        = fit(SmoothingSpline, 1/N:1/N:1, lognxs, .0001)
    lognxspred = predict(spl) # fitted vector
    for i in eachindex(nexpls)
        nexpls[i] = ceil(TI, exp(lognxspred[i]))
    end
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


