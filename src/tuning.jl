###############################################################################
# tuning routines
###############################################################################

#######################################
# main method
#######################################

# full tuning
function tune!(
    ns::NRSTSampler;
    max_rounds::Int    = 10,
    max_relΔlogZ::Real = 0.001,
    max_Δβs::Real      = 0.01,
    nsteps_init::Int   = 2000,
    max_nsteps::Int    = 65_536,
    nsteps_expl::Int   = 500,    # only used for tuning the explorers' params
    maxcor::Real       = 0.1,
    verbose::Bool      = true
    )
    N       = ns.np.N
    round   = 0
    nsteps  = nsteps_init
    oldlogZ = relΔlogZ = NaN # to assess convergence on the log(Z_N/Z_0) estimate
    aggV    = similar(ns.np.c)
    trVs    = [similar(aggV, nsteps) for _ in 0:N]
    conv    = false
    verbose && println("Tuning started ($(Threads.nthreads()) threads).")
    while !conv && (round < max_rounds)
        round += 1
        verbose && print("Round $round:\n\tTuning explorers...")
        tune_explorers!(ns, nsteps = nsteps_expl, verbose = false)
        verbose && println("done!")
        
        # tune c and betas
        verbose && print("\tTuning c and grid using $nsteps steps per explorer...")
        trVs     = [similar(aggV, nsteps) for _ in 0:N]
        Δβs      = tune_c_betas!(ns, trVs, aggV)
        relΔlogZ = abs(-ns.np.c[N+1] - oldlogZ) / abs(oldlogZ)
        oldlogZ  = -ns.np.c[N+1]
        verbose && @printf(
            "done!\n\t\tmax(Δbetas)=%.3f.\n\t\tlog(Z_N/Z_0)=%.1f.\n\t\trelΔlogZ=%.1f%%\n", 
            Δβs, -ns.np.c[N+1], 100*relΔlogZ
        )

        # check convergence
        conv = !isnan(relΔlogZ) && (relΔlogZ<max_relΔlogZ) && (Δβs<max_Δβs)
        nsteps = min(max_nsteps,2nsteps)
    end
    # since betas changed the last, need to tune explorers and c one last time
    verbose && print(
        (conv ? "Grid converged!" :
                "max_rounds=$max_rounds reached.") *
        "\nFinal round:\n\tTuning explorers..."
    )
    tune_explorers!(ns, nsteps = nsteps_expl, verbose = false)
    # nsteps = max(nsteps_max, 2^7 * nsteps_expl) # need high accuracy for final c estimate 
    # trVs = [similar(aggV, nsteps) for _ in 0:N]
    verbose && println("done!\n\tTuning c and nexpls using $(length(trVs[1])) steps per explorer...")
    collectVs!(ns, trVs, aggV)
    tune_c!(ns, trVs, aggV)
    tune_nexpls!(ns.np.nexpls, trVs, maxcor=maxcor)
    verbose && println("Tuning completed.")
end


#######################################
# under the hood
#######################################

# tune nexpls by imposing a threshold on autocorrelation
function tune_nexpls!(
    nexpls::Vector{TI},
    trVs::Vector{Vector{TF}};
    maxcor::AbstractFloat
    ) where {TI<:Int, TF<:AbstractFloat}
    # compute autocorrelations and build design matrix
    acs = [autocor(trV) for trV in trVs[2:end]];
    X   = reshape(collect(0:(length(acs[1])-1)),(length(acs[1]),1))
    L   = log(maxcor)
    for i in eachindex(nexpls)
        ac  = acs[i]
        idx = findfirst(x->isless(x,maxcor), ac) # attempt to find maxcor in acs
        if !isnothing(idx)
            nexpls[i] = idx - 1                  # acs starts at lag 0
        else                                     # extrapolate with model ac[n]=exp(ρn)
            y = log.(ac)
            ρ = (X \ y)[1]                       # solve least-squares: Xρ ≈ y
            nexpls[i] = ceil(L/ρ)                # x = e^{ρn} => log(x) = ρn => n = log(x)/ρ
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

# tune the explorers' parameters
function tune_explorers!(ns::NRSTSampler;smooth=true,kwargs...)
    Threads.@threads for expl in ns.explorers
        tune!(expl;kwargs...)
    end
    smooth && smooth_params!(ns.explorers,ns.np.betas)
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
    smooth && smooth_params!(ns.explorers,ns.np.betas)
end

# Tune c and betas using independent runs of the explorers
# note: explorers must be tuned already before running this
function tune_c_betas!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack c, betas = ns.np
    collectVs!(ns, trVs, aggV)         # in parallel, collect V samples and aggregate them
    @suppress_err tune_c!(ns,trVs,aggV)# tune c using aggV and trVs if necessary
    R = est_rej_probs(trVs, betas, c)  # compute average rejection probabilities
    oldbetas = copy(betas)             # store old betas to check convergence
    optimize_betas!(betas, R)          # tune using the inverse of Λ(β)
    reset_explorers!(ns)               # since betas changed, the cached potentials are stale
    return maximum(abs,betas-oldbetas) # for assessing convergence of the grid
end

# tune the c vector
function tune_c!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack use_mean, c, betas = ns.np
    if use_mean && (abs(aggV[1])>1e16)
        @info "V likely not integrable under the reference; using stepping stone."
        stepping_stone!(c, betas, trVs) # compute log(Z(b)/Z0) and store it in c
        c .*= (-one(K))                 # c(b) = -log(Z(b)/Z0)
    else
        trapez!(c, betas, aggV)         # trapezodial approximation of int_0^beta aggV(b)db
    end
    return
end

# utility to reset caches in all explorers
function reset_explorers!(ns::NRSTSampler)
    for e in ns.explorers
        set_state!(e, ns.x)
    end
end

# collect samples of V(x) at each of the levels, storing in trVs
# also compute V aggregate and store in aggV
function collectVs!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack fns, use_mean, N = ns.np
    @unpack V, randref = fns
    aggfun = use_mean ? mean : median
    nsteps = length(first(trVs)) 
    for i in 1:nsteps
        trVs[1][i] = V(randref())
    end
    aggV[1] = aggfun(trVs[1])
    Threads.@threads for i in 1:N         # "Threads.@threads for" does not work with enumerate(ns.explorers)
        ipone = i+1
        e = ns.explorers[i]
        run!(e, e.U.fns.V, trVs[ipone])   # Note: cant use shared np.V when threading. TODO: remove the cheap fix "e.U.fns.V" while fns becomes a proper field for explorers
        aggV[ipone] = aggfun(trVs[ipone])
    end
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
            R[i,1] = uno - mean([exp(-max(cero, dbs[i]*v-dcs[i])) for v in trV])
        end
        if i <= 1
            R[i,2] = uno
        else
            R[i,2] = uno - mean([exp(-max(cero, -dbs[i-1]*v+dcs[i-1])) for v in trV])
        end
    end
    return R
end

# optimize betas using the equirejection approach
function optimize_betas!(betas::Vector{K}, R::Matrix{K}) where {K<:AbstractFloat}
    # estimate Λ at current betas using rejections rates, normalize, and interpolate
    Λnorm, Λvalsnorm = get_lambda(betas, R) # note: this the only place where R is used

    # find newbetas by inverting Λnorm with a uniform grid on the range
    N           = length(betas) - 1
    Δ           = convert(K,1/N) # step size of the grid
    targetΛ     = zero(K)
    newbetas    = similar(betas)
    newbetas[1] = betas[1]       # technically 0., but is safer this way against rounding errors
    for i in 2:N
        targetΛ    += Δ
        b1          = newbetas[i-1]
        b2          = betas[findfirst(u -> (u>targetΛ), Λvalsnorm)]            # Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i] = find_zero(β -> Λnorm(β)-targetΛ, (b1,b2), atol = Δ*1e-4) # set tolerance for |Λnorm(β)-target| 
    end
    newbetas[end] = one(K)
    copyto!(betas, newbetas)
end

# estimate Λ at current betas using rejections rates, normalize, and interpolate
function get_lambda(betas::Vector{K}, R::Matrix{K}) where {K<:AbstractFloat}
    averej    = (R[1:(end-1),1] + R[2:end,2])/2 # average up and down rejections
    Λvals     = pushfirst!(cumsum(averej), 0.)
    Λvalsnorm = Λvals/Λvals[end]
    Λnorm     = interpolate(betas, Λvalsnorm, SteffenMonotonicInterpolation())
    
    # check error
    err = maximum(abs, Λnorm.(betas) - Λvalsnorm)
    if err > 10eps(K)
        @warn "get_lambda: high interpolation error = " err 
    end
    return (Λnorm, Λvalsnorm)
end


