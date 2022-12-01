###############################################################################
# Implements Sakai & Hukushima (2016), only the special case δ=1
###############################################################################

# exact same fields as NRSTSampler 
struct SH16Sampler{T,I<:Int,K<:AbstractFloat,TXp<:NRST.ExplorationKernel,TProb<:NRST.NRSTProblem} <: NRST.AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

# constructor: copy key fields of an existing (usually pre-tuned) NRSTSampler
SH16Sampler(ns::NRSTSampler) = SH16Sampler(NRST.copyfields(ns)...)

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

# Compute proposed i, neglog-prop-ratio (nlpr), and neglog-target-ratio (nltr). 
# This is eqs 12, 13, 14. Note: for δ=1 the proposals are deterministic
#     q_{0,1}^{eps} = q_{N,N-1}^{eps} = 1, at boundaries // this is true for all δ
#     q_{r,l}^{eps} = 1{l=r+eps},          o.w.
# Let
#     W_{r,l} := max{0,P_l/P_r} = exp{max{0,-nlar}}
# Then
#     W_{0,1}^{-} = 0                // since q_{1,0}^+ = 0 but q_{0,1}^-=1 (i.e., well-defined)
#     W_{N,N-1}^{+} = 0              // since q_{N-1,N}^- = 0 but q_{N,N-1}^+=1 (i.e., well-defined)
#     W_{r,l}^{eps} = W_{r,l}, o.w.
function propose_i(sh::SH16Sampler{T,I,K}) where {T,I,K}
    @unpack np,ip,curV = sh
    @unpack N,betas,c = np
    i    = first(ip)
    ϵ    = last(ip)
    nlpr = zero(K)
    if i == zero(I)
        iprop = one(I)
        ϵ < zero(I) && (nlpr = K(Inf))
    elseif i == N
        iprop = N-one(I)
        ϵ > zero(I) && (nlpr = K(Inf))
    else
        iprop = sum(ip)  # i+ϵ
    end        
    nltr = NRST.get_nlar(betas[i+1],betas[iprop+1],c[i+1],c[iprop+1],curV[])
    return iprop, nlpr, nltr
end

# Compute flip eps prob. This is Equations 15 & 16
# Note: for δ=1,
#    Λ_r^eps =  max{0, eps*S_r}
# where
#    S_r = sum_{l neq r}[q_{r,l}^{-}W_{r,l}^{-} - q_{r,l}^{+}W_{r,l}^{+}]
#        = {
#          -W_{0,1},          r=0
#          W_{N,N-1},         r=N
#          W_{r,r-1} - W_{r,r+1}, o.w.
# then we can write for 0<r<N
#    Λ_r^eps = max{0, W_{r,r-eps} - W_{r,r+eps} } 
# also
#    λ_r^eps = Λ_r^eps/D_r^eps
# with
#    D_r^eps = 1 - sum_{l neq r}q_{r,l}^{eps}W_{r,l}^{eps}
# = {
#    1-W_{0,1}^{eps},   r=0
#    1-W_{N,N-1}^{eps}, r=N
#    1-W_{r,r+eps},     o.w.
# finally
#    λ_r^eps = {
#     0,               (r,eps)=(0,+)  // Λ_0^+=max{0, S_0}=0 (S_0<=0)
#     W_{0,1},         (r,eps)=(0,-)  // W_{0,1}^{-} = 0 (see above function's comments)
#     W_{N,N-1},       (r,eps)=(N,+)  // W_{N,N-1}^{+} = 0 (same)
#     0,               (r,eps)=(N,-)  // Λ_N^-=-min{0, S_N}=0 (S_N>=0)
#     max{0, W_{r,r-eps} - W_{r,r+eps}} / [1 - W_{r,r+eps}], for 0<r<N // note this lies in [0,1]
function propose_flip(sh::SH16Sampler{T,I,K}, nltr) where {T,I,K}
    @unpack np,ip,curV = sh
    @unpack N,betas,c = np
    i = first(ip)
    ϵ = last(ip)
    nlar_ϵ = K(Inf)
    if (i == zero(I) && ϵ < zero(I)) || (i == N && ϵ > zero(I))
        nlar_ϵ = nltr
    elseif zero(I) < i && i < N
        apif = NRST.nlar_2_ap(nltr) # acc prob of i move in direction of eps (fwd)
        ibwd = i-ϵ
        apib = NRST.nlar_2_ap(      # acc prob of i move in opp direction of eps (bwd)
            NRST.get_nlar(betas[i+1],betas[ibwd+1],c[i+1],c[ibwd+1],curV[])
        )
        apib>apif && ( nlar_ϵ = log1p(-apif) - log(apib-apif) )
    end
    return nlar_ϵ
end

# both of the above
function propose(sh::SH16Sampler)
    iprop, nlpr, nltr = propose_i(sh)
    nlar_i = nlpr+nltr
    nlar_ϵ = propose_flip(sh, nltr) # we could avoid computing this for first case but good to have for checking 
    iprop, nlar_i, nlar_ϵ
end

# full tempering step. This is point (1) of the algorithm in Sec. 3.2.
function NRST.comm_step!(sh::SH16Sampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    iprop, nlar_i, nlar_ϵ = propose(sh)
    if nlar_i < randexp(rng)
        sh.ip[1] = iprop
    elseif nlar_ϵ < randexp(rng)
        sh.ip[2] *= -one(I)
    end
    return NRST.nlar_2_rp(nlar_ϵ)
end

# same step! method as NRST

#######################################
# RegenerativeSampler interface
#######################################

# same atom as NRST, no need for specialized isinatom method

# reset state by sampling from the renewal measure. Since
#     W_{0,1}^{-} = 0 
# we know for certain that the renewal measure only puts mass on (X,0,+1)
# Therefore, we can just use the same as for NRST
# function NRST.renew!

# handling last tour step
function NRST.save_last_step_tour!(sh::SH16Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(sh, tr; kwargs...)                       # store state at atom
    _, _, nlar_ϵ = propose(sh)                                   # simulate a proposal
    NRST.save_post_step!(sh, tr, NRST.nlar_2_rp(nlar_ϵ), K(NaN)) # save stats
end