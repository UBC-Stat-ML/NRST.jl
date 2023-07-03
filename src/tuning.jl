###############################################################################
# tuning routines
###############################################################################

# "optimal" value for N for a given Λ. γ is a multiplicative constant for 
# correcting the formula when asymptotic assumptions don't hold
optimal_N(Λ, γ) = ceil(Int, γ*Λ*(1+sqrt(1 + inv(1+2Λ))))
TE_inf(Λ) = inv(1+2Λ)

# full tuning: bootstraps using ensembles of explorers (NRPT or independent)
function tune!(
    ns::NRSTSampler,
    rng::AbstractRNG;
    use_TE_inf::Bool   = true,
    min_ntours::Int    = 2_048,
    verbose::Bool      = true,
    kwargs...
    )
    ens            = NRPTSampler(ns)                                # PT sampler, whose state is fully independent of ns
    oldsumnexpl    = sum(ns.np.nexpls)                              # store exploration cost of a full sweep of the states during tuning
    nsteps, Λ, lZs = tune!(ns.np, ens, rng;verbose=verbose,kwargs...)

    # estimate TE with preliminary NRST run: needed because we cannot estimate TE
    # with ensembles, since this is inherently a regenerative property.
    # two criteria for determining ntours necessary to estimate TE, take max
    # First: equal effort approach. Use same effort as last PT run:
    #   1 PT step = oldsumnexpl steps in the exploration kernels
    #   1 Tour    = 2sum(nexpl) steps in the exploration kernels
    # Second: match expected visits to top level under mean strategy. Require
    #   E[visits to N] = 2(p_N/p_0)ntours == 2min_ntours => ntours == min_ntours(p0/pN)
    # This defaults to min_ntours under mean strategy (that's why its called min_tours)
    p_ratio = exp((first(lZs)+first(ns.np.c)) - (last(lZs)+last(ns.np.c)))      # == p0/pN. no need to normalize because this is a ratio of probs
    @debug "tune!: p_ratio=$p_ratio"
    p_ratio < 0.001 && error("p0/pN<0.001 is too low ⟹ tuning failed! Increase γ or set 'use_mean=true' and try again.")
    ntours = min(DEFAULT_MAX_TOURS, use_TE_inf ? min_ntours_TE(TE_inf(Λ)/10) : # experimentally, TE ~ 20% TE_inf -> then add safety factor
        ceil(Int, max(
            nsteps * oldsumnexpl / max(1, 2*sum(ns.np.nexpls)),
            min_ntours*(p_ratio^1.3)                                # exponent is a safety factor
        ))
    )
    verbose && println("\nEstimating Tour Effectiveness (TE) using $ntours NRST tours.\n")
    res = parallel_run(ns, rng; ntours = ntours, verbose = verbose)
    TE  = last(res.toureff)
    @printf("\nTuning completed! Summary:\n\tTE = %.2f\n\tΛ  = %.2f\n",TE, Λ)
    return TE, Λ
end

# stage I tuning: bootstrap ensembles of explorers
function tune!(
    np::NRSTProblem,
    ens::NRPTSampler,
    rng::AbstractRNG;
    max_rounds::Int    = 14,
    max_ar_ratio::Real = 0.10,      # limit on std(ar)/mean(ar), ar: average of Ru and Rd, the directional rejection rates
    max_dr_ratio::Real = 0.05,      # limit on mean(|Ru-Rd|)/mean(ar). Note: this only makes sense for use_mean=true
    max_relΔcone::Real = 0.005,     # limit on rel change in c(1)
    max_relΔΛ::Real    = 0.01,      # limit on rel change in Λ = Λ(1)
    nsteps_init::Int   = 2,         # steps used in the first round
    maxcor::Real       = 0.95,      # set nexpl in explorers s.t. correlation of V samples is lower than this
    γ::Real            = 2.0,       # correction for the optimal_N formula
    xpl_smooth_λ::Real = 1e-5,      # smoothness knob for xpl params. λ==0 == no smoothing
    check_N::Bool      = true,
    check_at_rnd::Int  = 9,         # early round with enough accuracy to check V integrability and N 
    adapt_nexpls::Bool = true,      # should we adapt number of exploration steps after the last round?
    verbose::Bool,
    kwargs...
    )
    !np.use_mean && (max_dr_ratio = Inf)      # equality of directional rejections only holds for the mean strategy
    if verbose
        println(
            "Tuning started ($(Threads.nthreads()) threads).\n\n" *
            "Bootstrapping NRST using NRPT.\n"
        )
        show(plot_grid(np.betas, title="Histogram of βs: initial grid"))
        println("\n")
    end
    rnd     = 0
    nsteps  = nsteps_init÷2                 # nsteps is doubled at the beginning of the loop
    oldcone = relΔcone = oldΛ = relΔΛ = NaN
    conv    = false
    while !conv && (rnd < max_rounds)
        rnd    += 1
        nsteps *= 2        
        verbose && println("Round $rnd:")
        
        # tune c and betas
        verbose && print("\tTuning explorers, c, and grid using $nsteps steps...")
        res          = @timed tune_inner!(np, ens, rng, nsteps)
        ξ,Δβs,Λ,ar,R = res.value # note: rejections are before grid adjustment, so they are technically stale, but are still useful to assess convergence. compute std dev of average of up and down rejs
        mar          = mean(ar)
        ar_ratio     = std(ar)/mar
        ar1_ratio    = ar[1]/mean(ar[2:end])
        dr_ratio     = mean(abs, R[1:(end-1),1] - R[2:end,2])/mar
        relΔcone     = abs(np.c[end] - oldcone) / abs(oldcone)
        oldcone      = np.c[end]
        relΔΛ        = abs(Λ - oldΛ) / abs(oldΛ)
        oldΛ         = Λ
        if verbose 
            @printf(
                """
                done!
                \t\tAR std/mean=%.3f
                \t\tAR[1]/mean(AR[-1])=%.3f
                \t\tξ(|V| @ i=0)=%.2f
                \t\tmean|Ru-Rd|/mean=%.3f
                \t\tmax(Δbetas)=%.3f
                \t\tc(1)=%.2f (relΔ=%.2f%%)
                \t\tΛ=%.2f (relΔ=%.1f%%)
                \t\tElapsed: %.1fs\n\n 
                """, ar_ratio, ar1_ratio, ξ, dr_ratio, Δβs, last(np.c), 
                100*relΔcone, Λ, 100*relΔΛ, res.time
            )
            show(plot_grid(np.betas, title="Histogram of βs: round $rnd"))
            println("\n")
        end

        # time to check if parameters are ok
        if rnd == check_at_rnd            
            if !np.log_grid && ξ >= 1                                     # if E_0[|V|]=∞, we need to switch interpolating Λ(β) in log scale to handle Inf derivative at 0.
                throw(NonIntegrableVException())
            end
            if check_N                                                    # check if N is too low / high
                Nopt = optimal_N(Λ, γ)
                dN   = abs(np.N-Nopt)
                dN > 1 && dN > .1*Nopt && throw(BadNException(np.N,Nopt))
            end
        end

        # check convergence
        conv = (rnd >= check_at_rnd) && !isnan(relΔcone) && 
            (relΔcone<max_relΔcone) && !isnan(relΔΛ) && (relΔΛ < max_relΔΛ) &&
            (ar_ratio < max_ar_ratio) && (dr_ratio < max_dr_ratio)
    end

    # at this point, ns has a fresh new grid, so explorers and c are stale
    # => we need to tune them.
    if verbose 
        println(conv ? "Grid converged!" : "max_rounds=$max_rounds reached.")
        println("\nAdjusting settings to new grid:")
        print("\tTuning explorers and c using $nsteps steps...")      
    end
    res = @timed tune_last!(np, ens, rng, nsteps)
    Λ, lZs = res.value
    verbose && @printf("done!\n\t\tElapsed: %.1fs\n", res.time)

    # if requested, adapt the number of exploration steps
    if adapt_nexpls
        verbose && print("\tTuning nexpls using $nsteps steps...")      
        res = @timed tune_nexpls!(np, ens, rng, nsteps, maxcor, xpl_smooth_λ)
        if verbose
            @printf("done!\n\t\tElapsed: %.1fs\n\n", res.time)
            show(
                lineplot_term(
                    np.betas[2:end], np.nexpls, xlabel = "β",
                    title="Exploration steps needed to get correlation ≤ $maxcor"
                )
            ); println("\n")
        end
    else
        println()
    end

    # after these steps, NRST is coherently tuned and can be used to sample
    return nsteps, Λ, lZs
end

# tune explorers, c, and betas. also calculate tail index of |V| at level 0
# note the order: first tune c, then grid. This because c is used for 
# estimating NRST rejections, which are then used to estimate Lambda
function tune_inner!(np::NRSTProblem, args...)
    # print("tuning xpls and c...")
    trVs = tune_xpls_and_c!(np, args...)
    # print("done!\nFitting GPD to V tail...")
    ξ    = gpd_index(first(trVs))
    # print("done!\nOptimizing grid...")
    res  = tune_betas!(np, trVs)
    # println("done!")
    (ξ, res...)
end

# last round of tuning after last adjustment to the grid
# tune explorers, c, and number of exploration steps
function tune_last!(
    np::NRSTProblem,
    nrpt::NRPTSampler,
    rng::AbstractRNG,
    nsteps::Int
    )
    ptVs = tune_xpls_and_c!(np, nrpt, rng, nsteps)
    Λ    = last(get_lambdas(averej(est_rej_probs(ptVs, np.betas, np.c))))     # final estimate of Λ using the most recent grid and c values
    lZs  = log_partition(np, ptVs)                                            # estimate log(Z_i)
    return Λ, lZs
end

# tune explorers (TODO!) and c using samples from NRPT
function tune_xpls_and_c!(np::NRSTProblem, nrpt::NRPTSampler, args...)
    tr   = run!(nrpt, args...)
    # TODO: pass tr.xs to explorers' tuning method
    trVs = rows2vov(tr.Vs)
    tune_c!(np, trVs)
    return trVs
end

##############################################################################
# under the hood
##############################################################################

##############################################################################
# estimate tail index of |V| at level 0
##############################################################################

function gpd_index(V0s::Vector{<:AbstractFloat})
    if length(V0s) > 30                                                          # only do this for sufficiently large sample
        as   = abs.(V0s)
        as .-= minimum(as)                                                       # center because gpd_fit assumes location=0
        ξ    = first(ParetoSmooth.gpd_fit(as, 1.0, wip=false, sort_sample=true)) # 1.0 eff because this is iid sampling. No prior because it defaults to ~0.5 and that would be optimistic when testing <1
    else
        ξ    = eltype(V0s)(NaN)
    end
    return ξ
end

##############################################################################
# tune c
##############################################################################

# tune the c vector using samples of the potential function
function tune_c!(np::NRSTProblem{T,K}, trVs::Vector{Vector{K}}) where {T,K}
    if np.use_mean                                    # use mean strategy => can use thermo. ident. to use stepping stone, which is simulation consistent
        stepping_stone!(np.c, np.betas, trVs)         # compute log(Z(b)/Z0) and store it in c
        np.c .*= (-one(K))                            # c(b) = -log(Z(b)/Z0)
    else
        trapez!(np.c, np.betas, median.(trVs))        # use trapezoidal approx of int_0^beta med(b)db
    end
    return
end

##############################################################################
# tune betas
##############################################################################

# uses the average of the up and down rates of rejection
function tune_betas!(np::NRSTProblem, ar::Vector{<:Real})
    betas    = np.betas
    oldbetas = copy(betas)                            # store old betas to check convergence
    _, _, Λs = optimize_grid!(betas, ar, np.log_grid) # tune using the inverse of Λ(β)
    return (maximum(abs,betas-oldbetas),Λs[end])      # return max change in grid and Λ=Λ(1) to check convergence
end

# method for the raw traces of V along the levels
function tune_betas!(np::NRSTProblem{T,K}, trVs::Vector{Vector{K}}) where {T,K}
    R   = est_rej_probs(trVs, np.betas, np.c)         # compute average directional NRST rejection probabilities
    ar  = averej(R)                                   # average up and down directions
    out = tune_betas!(np, ar)
    (out..., ar, R)
end

#######################################
# methods to estimate rejections
#######################################

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

#######################################
# working with the Lambda function
#######################################

floorlog(x) = max(LOGSMALL, log(x))
expfloor(x) = (x <= LOGSMALL ? zero(x) : exp(x))

# optimize the grid using the equirejection approach
# Note: need to work in log space whenever
#   (dΛ/dβ)(0) = E^{0}[|V - dc/dβ|] = ∞
# which happens when V is not integrable at the reference; i.e., when
#   KL(ref|target) = ∞
function optimize_grid!(betas::Vector{K}, averej::Vector{K}, uselog::Bool) where {K<:AbstractFloat}
    # estimate Λ at current betas using rejections rates, normalize, and interpolate
    f_Λnorm, Λsnorm, Λs = gen_lambda_fun(betas, averej, uselog) # note: this the only place where averej is used
    
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
        # @debug "optimize_grid: looking for β in ($b1,$b2) to match f(β)=$targetΛ"
        newbetas[i] = optimize_grid_core(f_Λnorm, targetΛ, b1, b2, uselog)
        newbetas[i] == b1 && 
            error(
                "optimize_grid: got equal consecutive beta[$i]=beta[$(i-1)]=$b1.\n" *
                (i != 2 ? "" : "Potential reason: π₀{V=inf}>0. Try setting " *
                               "reject_big_vs=true."
                )
            )
        # @debug "optimize_grid: found β=$(newbetas[i])"
    end
    newbetas[end] = one(K)
    copyto!(betas, newbetas)
    return (f_Λnorm, Λsnorm, Λs)
end

# estimate Λ, normalize, and interpolate
function gen_lambda_fun(betas::Vector{K}, averej::Vector{K}, uselog::Bool) where {K<:AbstractFloat}
    Λs      = get_lambdas(averej)
    Λsnorm  = Λs/Λs[end]
    xs      = uselog ? floorlog.(betas) : betas
    f_Λnorm = interpolate(xs, Λsnorm, FritschButlandMonotonicInterpolation())
    
    # check fit
    fitΛs  = map(f_Λnorm, xs)
    idxnan = findfirst(isnan, fitΛs)
    isnothing(idxnan) || error("gen_lambda_fun: interpolated f produces NaN at " *
                               "(i,β)=($idxnan,$(betas[idxnan]))")
    err = mapreduce((f,y)-> abs(y-f), max, fitΛs, Λsnorm)
    err > 10eps(K) && @warn "gen_lambda_fun: high interpolation error = " err
    return (f_Λnorm, Λsnorm, Λs)
end

# solve Λ(β) = targetΛ, possibly in log space
function optimize_grid_core(f_Λnorm, targetΛ, b1, b2, uselog)
    if uselog
        lβ, _ = monoroot(lβ -> f_Λnorm(lβ)-targetΛ, floorlog(b1), floorlog(b2))
        return expfloor(lβ)
    else
        return first(monoroot(β -> f_Λnorm(β)-targetΛ, b1, b2))
    end
end

##############################################################################
# tune nexpls by imposing a threshold on autocorrelation
##############################################################################

# take serial sample with NRPT by skipping communication, then pass it to core method
function tune_nexpls!(np::NRSTProblem, nrpt::NRPTSampler, rng, nsteps, args...)
    tune_nexpls!(np.nexpls, collectVsSerial!(nrpt, rng, nsteps), args...)
end

# core method
function tune_nexpls!(
    nexpls::Vector{TI},
    trVs::Vector{Vector{TF}},
    maxcor::AbstractFloat,
    λ::AbstractFloat = 0.;
    maxTF::TF = inv(eps(TF))
    ) where {TI<:Int, TF<:AbstractFloat}
    N       = length(nexpls)
    L       = log(maxcor)
    idxfail = TI[]
    for i in eachindex(nexpls)
        # sanity checks of V samples
        trV = clamp.(trVs[i+1], -maxTF, maxTF) # clamp needed to avoid stddev = NaN => autocor=NaN
        if all(v -> v==first(trV), trV)
            @debug "tune_nexpls: explorer $i produced constant V samples; skipping it."
            push!(idxfail, i)
            nexpls[i] = -one(TI)
            continue
        end

        # compute autocorrelations and try finding something smaller than maxcor
        ac  = autocor(trV)
        idx = findfirst(a -> a<=maxcor, ac)
        if !isnothing(idx)
            nexpls[i] = idx - one(TI)            # acs starts at lag 0
        else                                     # extrapolate with model ac[n]=exp(ρn)
            l  = length(ac)
            xs = 0:(l-1)
            ys = log.(ac)
            ρ  = sum(xs .* ys) / sum(abs2, xs)   # solve least-squares: y ≈ ρx
            nexpls[i] = ceil(TI, L/ρ)            # c = e^{ρn} => log(c) = ρn => n = log(c)/ρ
        end
    end

    # interpolate any element that failed
    if length(idxfail) > 0
        @debug "tune_nexpls: fixing errors at $idxfail by interpolation."
        idxgood = setdiff(1:N, idxfail)
        xs  = idxgood ./ N
        ys  = log.(nexpls[idxgood])
        itp = linear_interpolation(xs, ys, extrapolation_bc=Line())
        for i in idxfail
            nexpls[i] = ceil(TI, exp(itp(i/N)))
        end
    end

    # minimal smoothing to remove extreme outliers caused by std(trV) ≈ 0
    # due to low acceptance rate (i.e., when in a corner of the state space) 
    if λ > 0.
        w = closest_odd(λ*N) 
        snexpls = running_median(nexpls, w, :asymmetric_truncated) # asymmetric_truncated also smooths endpoints
        for (i, s) in enumerate(snexpls)
            nexpls[i] = ceil(TI, s) 
        end
    end
    return
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
