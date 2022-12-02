###############################################################################
# Implements Faizi et al. (2020) with Irreversible Metropolized-Gibbs sampler.
# Only the special case δ=1, shown by authors to outperform others
###############################################################################

# exact same fields as NRSTSampler, plus storage for calculations
struct FBDRSampler{T,I<:Int,K<:AbstractFloat,TXp<:NRST.ExplorationKernel,TProb<:NRST.NRSTProblem} <: NRST.AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    gs::Vector{K}          # storage for the log of the conditional beta|x, length N+1
    ms::Vector{K}          # storage for the log of the Metropolized Gibbs conditionals, length N+1
end

# constructor: copy key fields of an existing (usually pre-tuned) NRSTSampler
function FBDRSampler(ns::NRSTSampler)
    FBDRSampler(NRST.copyfields(ns)...,similar(ns.np.c),similar(ns.np.c))
end

# specialized copy method to deal with extra storage fields
function Base.copy(fbdr::TS) where {TS <: FBDRSampler} 
    TS(NRST.copyfields(fbdr)...,similar(fbdr.np.c),similar(fbdr.np.c))
end

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

# update the log of the Gibbs conditional of beta
#     G_i = exp(-b_iV + c_i)/ sum_j exp(-b_jV + c_j)
#     g_i := log(G_i) = -b_iV + c_i - logsumexp(-bV + c)
# note: it holds that sum(exp, gs) === one(eltype(gs))
function update_gs!(fbdr::FBDRSampler{T,I,K}) where {T,I,K}
    @unpack gs,np,curV = fbdr
    @unpack betas,c = np
    gs  .= -betas .* curV[] .+ c
    gs .-= logsumexp(gs)
end

# update the log of the irreversible Metropolized Gibbs (IMGS) conditional of beta
# the Irreversible Metropolized conditional is (Eq 31)
#     M_{i,j}^{eps} =
#     {
#      M_{i,j},                 (j-i)eps>0
#      0,                       (j-i)eps<0
#      1-sum_{j neq i} M_{i,j}, j==i        
# with M_{i,j} the (standard) Metropolized Gibbs (Eq 15)
#     M_{i,j} = G_j min{1/(1-G_i), 1/(1-G_j)}, j neq i
# so
#     m_j := log(M_{i,j}^{eps}) =
#     {
#      log(M_{i,j}),                 (j-i)eps>0
#      -Inf,                         (j-i)eps<0
#      log(1-sum_{j neq i} M_{i,j}), j==i      
# Furthermore  
#     m_j = g_j - max{log1mexp(g_i),log1mexp(g_j)}, (j-i)eps>0
#     m_i = log(1-sum(M_{i,j})) = log(1-exp(logsumexp(m))) = log1mexp(logsumexp(m))
# note: it holds that sum(exp, ms) === one(eltype(ms))
function update_ms!(fbdr::FBDRSampler{T,I,K}) where {T,I,K}
    update_gs!(fbdr)
    @unpack gs,ms,ip = fbdr
    i   = first(ip)
    idx = i+one(I)
    ϵ   = last(ip)
    log1mexpgi = log1mexp(gs[idx])
    for (jdx,g) in enumerate(gs)
        # for j=i we also put -Inf so that it contributes 0 to the logsumexp below
        @inbounds ms[jdx] = sign(jdx-idx)!=sign(ϵ) ? K(-Inf) : g - max(log1mexp(g), log1mexpgi)
    end
    ms[idx] = log1mexp(min(zero(K), logsumexp(ms))) # truncation is needed to handle numerical errors
end

# full tempering step
function NRST.comm_step!(fbdr::FBDRSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack ms,ip,np = fbdr
    i = first(ip)                                         # attempt to move i
    update_ms!(fbdr)                                      # update IMGS probabilities
    iprop   = sample_logprob(rng, ms) - one(I)            # sample a new i (1-based index)
    lpstayf = ms[i+1]                                     # log-probability of iprop==i
    if iprop != i
        ip[1] = iprop 
    else                                                  # got same i, so need to attempt flip
        ip[2]  *= -one(I)                                 # simulate flip
        update_ms!(fbdr)                                  # recompute IMGS probabilities
        lpstayb = ms[i+1]                                 # log-probability of iprop==i with flip
        lpflip  = log1mexp(min(zero(K), lpstayb-lpstayf)) # log-probability of flip
        randexp(rng) < -lpflip && (ip[2] *= -one(I))      # flip failed => need to undo ϵ flip   
    end
    return exp(lpstayf)                                   # lpstayf == logprob of rejecting an i move
end

# same step! method as NRST
# note: the following should always return true
# NRST.step!(fbdr, rng)
# NRST.CompetingSamplers.update_ms!(fbdr)
# first(ip) == (last(ip) > 0 ? findfirst(isfinite,ms) : findlast(isfinite,ms))-1

#######################################
# RegenerativeSampler interface
#######################################

# same atom as NRST, no need for specialized isinatom method

# reset state by sampling from the renewal measure
# not sure if P((0,-1)->(0,1))=1 (experimentally it looks like this is true),
# so safer to have a specialized method
function NRST.renew!(fbdr::FBDRSampler{T,I}, rng::AbstractRNG) where {T,I}
    fbdr.ip[1] = zero(I)
    fbdr.ip[2] = -one(I)
    NRST.step!(fbdr, rng)
end

# handling last tour step
function NRST.save_last_step_tour!(fbdr::FBDRSampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(fbdr, tr; kwargs...)   # store state at atom
    update_ms!(fbdr)                           # update IMGS probabilities
    rp = exp(fbdr.ms[first(fbdr.ip)+1])        # probability of iprop==i <=> prob of rejecting an i move
    NRST.save_post_step!(fbdr, tr, rp, K(NaN)) # save stats
end
