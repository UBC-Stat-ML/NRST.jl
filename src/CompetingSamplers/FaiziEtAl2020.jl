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

# update the log of the irreversible Metropolized Gibbs conditional of beta
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
#     m_j = log(M_{i,j}) = g_j - max{log1mexp(g_i),log1mexp(g_j)}, (j-i)eps>0
#     m_i = log(M_{i,i}^{eps}) = log(1-sum(M_{i,j})) = log(1-exp(logsumexp(m))) = log1mexp(logsumexp(m))
# note: it holds that sum(exp, ms) === one(eltype(ms))
function update_ms!(fbdr::FBDRSampler{T,I,K}) where {T,I,K}
    update_gs!(fbdr)
    @unpack gs,ms,ip = fbdr
    i = first(ip)
    ϵ = last(ip)
    log1mexpgi = log1mexp(gs[i])
    for (j,g) in enumerate(gs)
        # for j=i we also put -Inf so that it contributes 0 to the logsumexp below
        @inbounds ms[j] = sign(j-i)!=sign(ϵ) ? K(-Inf) : g - max(log1mexp(g), log1mexpgi)
    end
    ms[i] = log1mexp(min(zero(K), logsumexp(ms))) # truncation is needed to handle numerical errors
end

# full tempering step
function NRST.comm_step!(fbdr::FBDRSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack ms,ip,np = fbdr
    N = np.N
    i = first(ip)
    ϵ = last(ip)

    # attempt i move
    update_ms!(fbdr)
    iprop = sample_logprob(rng, ms) - one(I)
    if iprop != i
        ip[1] = iprop
    elseif
    end
    
    iprop, nlar_i, nlar_ϵ = propose(sh)
    if nlar_i < randexp(rng)
        sh.ip[1] = iprop
    elseif nlar_ϵ < randexp(rng)
        sh.ip[2] *= -one(I)
    end
    return NRST.nlar_2_rp(nlar_ϵ)
end

# same step! method as NRST

# #######################################
# # RegenerativeSampler interface
# #######################################

# # same atom as NRST, no need for specialized isinatom method

# # reset state by sampling from the renewal measure. Since
# #     W_{0,1}^{-} = 0 
# # we know for certain that the renewal measure only puts mass on (X,0,+1)
# # Therefore, we can just use the same as for NRST
# # function NRST.renew!

# # handling last tour step
# function NRST.save_last_step_tour!(sh::SH16Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
#     NRST.save_pre_step!(sh, tr; kwargs...)                       # store state at atom
#     _, _, nlar_ϵ = propose(sh)                                   # simulate a proposal
#     NRST.save_post_step!(sh, tr, NRST.nlar_2_rp(nlar_ϵ), K(NaN)) # save stats
# end
