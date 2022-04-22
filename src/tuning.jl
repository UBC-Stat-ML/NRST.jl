###############################################################################
# tuning routines
###############################################################################

#######################################
# main method
#######################################

# full tuning
function tune!(
    ns::NRSTSampler;
    max_rounds::Int      = 16,
    max_chng_thrsh::Real = 0.01,
    nsteps_expl::Int     = max(500, 10*ns.nexpl),
    nsteps_max::Int      = 100_000,
    verbose::Bool        = true
    )
    N        = ns.np.N
    round    = 0
    nsteps   = nsteps_expl
    max_chng = 1.
    aggV     = similar(ns.np.c)
    
    verbose && println("Tuning started ($(Threads.nthreads()) threads).")
    while (max_chng > max_chng_thrsh) && (round < max_rounds)
        round += 1
        verbose && print("Round $round:\n\tTuning explorers...")
        tune_explorers!(ns, nsteps = nsteps_expl, verbose = false)
        verbose && println("done!")
        
        # tune c and betas
        verbose && print("\tTuning c and grid using $nsteps steps per explorer...")
        trVs     = [similar(aggV, nsteps) for _ in 0:N]
        max_chng = tune_c_betas!(ns, trVs, aggV)
        verbose && @printf("done!\n\t\tGrid change Δmax=%.3f.\n", max_chng)
        nsteps   = min(nsteps_max, 2nsteps)
    end
    # since betas changed the last, need to tune explorers and c one last time
    verbose && print(
        (max_chng <= max_chng_thrsh ? "Grid converged!" : 
                                      "max_rounds=$max_rounds reached.") *
        "\nFinal round:\n\tTuning explorers..."
    )
    tune_explorers!(ns, nsteps = nsteps_expl, verbose = false)
    verbose && print("done!\n\tTuning c using $nsteps steps per explorer...")
    collectVs!(ns, [similar(aggV, nsteps) for _ in 0:N], aggV)
    trapez!(ns.np.c, ns.np.betas, aggV)
    verbose && println("done!\nTuning finished.")
end


#######################################
# under the hood
#######################################

# tune the explorers' parameters
function tune_explorers!(ns::NRSTSampler;kwargs...)
    Threads.@threads for expl in ns.explorers
        tune!(expl;kwargs...)
    end
end

# tune the explorers' parameters serially. can use previous expl's params as
# warm start for the next.
function tune_explorers_serial!(ns::NRSTSampler;kwargs...)
    tune!(ns.explorers[1];kwargs...)
    for i in 2:ns.np.N
        # use previous explorer's params as warm start
        pars = params(ns.explorers[i-1])
        tune!(ns.explorers[i], pars;kwargs...)
    end
end

# Tune c and betas using independent runs of the explorers
# note: explorers must be tuned already before running this
# - idea:
#    - in parallel, assign each explorer to
#       - gather estimates of V and
#       - estimate up and down swap Metropolis ratio, sans the c portion
#    - after exploration, compute c from the V estimates
#    - add the c portion to the swap Metropolis ratios to obtain rej probs
#    - estimate Λ(β) and adjust the grid
function tune_c_betas!(
    ns::NRSTSampler{T,I,K},
    trVs::Vector{Vector{K}},
    aggV::Vector{K}
    ) where {T,I,K}
    @unpack c, betas = ns.np
    collectVs!(ns, trVs, aggV)        # collect V samples and aggregate them
    trapez!(c, betas, aggV)           # store in c the trapezoidal approx of int_0^beta db aggV(b)
    R = est_rej_probs(trVs, betas, c) # compute average rejection probabilities
    oldbetas = copy(betas)            # store old betas to check convergence
    optimize_betas!(betas, R)         # tune using the inverse of Λ(β)
    reset_explorers!(ns)              # since betas changed, the cached potentials are stale
    return maximum(abs,betas-oldbetas)# for assessing convergence of the grid
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
    N           = size(R, 1) - 1
    Δ           = convert(K,1/N) # step size of the grid
    targetΛ     = zero(K)
    newbetas    = similar(betas)
    newbetas[1] = betas[1]       # technically 0., but is safer this way against rounding errors
    for i in 2:N
        targetΛ    += Δ
        b1          = newbetas[i-1]
        b2          = betas[findfirst(u -> (u>targetΛ), Λvalsnorm)]            # Λnorm^{-1}(targetΛ) cannot exceed this
        newbetas[i] = find_zero(β -> Λnorm(β)-targetΛ, (b1,b2), atol = 0.01*Δ) # set tolerance for |Λnorm(β)-target| 
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
    @assert sum(abs, Λnorm.(betas) - Λvalsnorm) < 10eps(K)
    return (Λnorm, Λvalsnorm)
end


