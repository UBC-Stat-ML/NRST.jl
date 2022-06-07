##############################################################################
# raw trace of a serial run
##############################################################################

struct NRSTTrace{T,TI<:Int,TF<:AbstractFloat}
    trX::Vector{T}
    trIP::Vector{SVector{2,TI}}
    trV::Vector{TF}
    trRP::Vector{TF}
    trXplAP::Vector{Vector{TF}}
end

# outer constructors that allocate empty arrays with conservative sizehint
function NRSTTrace(::Type{T}, N::TI,::Type{TF}) where {T,TI<:Int,TF<:AbstractFloat}
    M       = 20N                                          # simple conservative estimate = 10E[T]
    trX     = sizehint!(T[], M)
    trIP    = sizehint!(SVector{2,TI}[], M)                # can use a vector of SVectors since traces should not be modified
    trV     = sizehint!(TF[], M)
    trRP    = sizehint!(TF[], M)
    trXplAP = [sizehint!(TF[], M) for _ in 1:N]
    NRSTTrace(trX, trIP, trV, trRP, trXplAP)
end
function NRSTTrace(::Type{T}, N::TI,::Type{TF}, nsteps::Int) where {T,TI<:Int,TF<:AbstractFloat}
    trX     = Vector{T}(undef, nsteps)
    trIP    = Vector{SVector{2,TI}}(undef, nsteps) # can use a vector of SVectors since traces should not be modified
    trV     = Vector{TF}(undef, nsteps)
    trRP    = similar(trV)
    trXplAP = [TF[] for _ in 1:N]                  # cant predict number of visits to each level
    NRSTTrace(trX, trIP, trV, trRP, trXplAP)
end
get_N(tr::NRSTTrace) = length(tr.trXplAP)  # recover N. cant do N=N(tr) because julia gets dizzy
get_nsteps(tr::NRSTTrace) = length(tr.trV) # recover nsteps
# NRSTTrace(tr::NRSTTrace{T,TI,TF}) where {T,TI,TF} = NRSTTrace(T, get_N(tr), TF) # construct empty trace of the same type that another
# function Base.empty!(tr::NRSTTrace{T,TI,TF}) where {T,TI,TF}
#     empty!(tr.trX)
#     empty!(tr.trIP)
#     empty!(tr.trV)
#     empty!(tr.trRP)
#     empty!.(tr.trXplAP) # empty the component vectors not the outer vector, so that it retains length N
#     return
# end
# function Base.resize!(tr::NRSTTrace{T,TI,TF}, n::Int) where {T,TI,TF}
#     resize!(tr.trX, n)
#     resize!(tr.trIP, n)
#     resize!(tr.trV, n)
#     resize!(tr.trRP, n)
#     return
# end
# function Base.copy(tr::NRSTTrace{T,TI,TF}) where {T,TI,TF}
#     NRSTTrace(
#         copy(tr.trX),
#         copy(tr.trIP),
#         copy(tr.trV),
#         copy(tr.trRP),
#         copy.(tr.trXplAP)
#     )
# end

#######################################
# trace postprocessing
#######################################

abstract type RunResults{T,TI<:Int,TF<:AbstractFloat} end

get_N(res::RunResults) = length(res.trVs)-1 # retrieve max tempering level

#######################################
# serial
#######################################

struct SerialRunResults{T,TI,TF} <: RunResults{T,TI,TF}
    tr::NRSTTrace{T,TI}       # raw trace
    xarray::Vector{Vector{T}} # i-th entry has samples at state i
    trVs::Vector{Vector{TF}}  # i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}        # total number of visits to each (i,eps)
    rpacc::Matrix{TF}         # accumulates rejection probs of swaps started from each (i,eps)
    xplapac::Vector{TF}       # length N. accumulates explorers' acc probs
end

# outer constructor that parses a trace
function SerialRunResults(tr::NRSTTrace{T,I,K}) where {T,I,K}
    N       = get_N(tr)
    xarray  = [T[] for _ in 0:N] # i-th entry has samples at state i
    trVs    = [K[] for _ in 0:N] # i-th entry has Vs corresponding to xarray[i]
    visacc  = zeros(I, N+1, 2)   # accumulates visits
    rpacc   = zeros(K, N+1, 2)   # accumulates rejection probs
    xplapac = zeros(K, N)        # accumulates explorers' acc probs
    post_process(tr, xarray, trVs, visacc, rpacc, xplapac)
    SerialRunResults(tr, xarray, trVs, visacc, rpacc, xplapac)
end
function post_process(
    tr::NRSTTrace{T,I,K},
    xarray::Vector{Vector{T}}, # length = N+1. i-th entry has samples at state i
    trVs::Vector{Vector{K}},   # length = N+1. i-th entry has Vs corresponding to xarray[i]
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rpacc::Matrix{K},          # size (N+1) × 2. accumulates rejection probs
    xplapac::Vector{K}         # length = N. accumulates explorers' acc probs
    ) where {T,I,K}
    for (n, ip) in enumerate(tr.trIP)
        l      = ip[1]
        idx    = l + 1
        idxeps = (ip[2] == one(I) ? 1 : 2)
        visacc[idx, idxeps]  += one(I)
        rpacc[idx, idxeps]   += tr.trRP[n]
        if l >= 1
            nvl = visacc[idx,1] + visacc[idx,2] # (>=1) number of visits so far to level l, regardless of eps
            xplapac[l] += tr.trXplAP[l][nvl]
        end
        length(tr.trX) >= n && push!(xarray[idx], tr.trX[n]) # handle case keep_xs=false
        push!(trVs[idx], tr.trV[n])
    end
end

#######################################
# touring
#######################################

struct TouringRunResults{T,TI,TF} <: RunResults{T,TI,TF}
    trvec::Vector{NRSTTrace{T,TI,TF}} # vector of raw traces from each tour
    xarray::Vector{Vector{T}}               # length = N+1. i-th entry has samples at level i
    trVs::Vector{Vector{TF}}                # length = N+1. i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}                      # total number of visits to each (i,eps)
    rpacc::Matrix{TF}                       # accumulates rejection probs of swaps started from each (i,eps)
    xplapac::Vector{TF}                     # length N. accumulates explorers' acc probs
    toureff::Vector{TF}                     # tour effectiveness for each i ∈ 0:N
end
get_ntours(res::TouringRunResults) = length(res.trvec)
tourlengths(res::TouringRunResults) = get_nsteps.(res.trvec)

# outer constructor that parses a vector of serial traces
function TouringRunResults(results::Vector{TST}) where {T,I,K,TST<:NRSTTrace{T,I,K}}
    ntours  = length(results)
    N       = get_N(first(results))
    xarray  = [T[] for _ in 0:N]         # i-th entry has samples at level i
    trVs    = [K[] for _ in 0:N]         # i-th entry has Vs corresponding to xarray[i]
    curvis  = Matrix{I}(undef, N+1, 2)   # visits in current tour to each (i,eps) state
    sumsq   = zeros(I, N+1)              # accumulate squared number of visits for each 0:N state (for tour effectiveness)
    totvis  = zeros(I, N+1, 2)           # total visits to each (i,eps) state
    rpacc   = zeros(K, N+1, 2)           # accumulates rejection probs of swaps started from each (i,eps)
    xplapac = zeros(K, N)                # accumulates explorers' acc probs
    
    # iterate tours
    for (_, tr) in enumerate(results)
        fill!(curvis, zero(I))                                 # reset tour visits
        post_process(tr, xarray, trVs, curvis, rpacc, xplapac) # parse tour trace
        totvis .+= curvis                                      # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2                 # squared number of visits to each of 0:N (regardless of direction)
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq)    # = (sum(totvis, dims=2)/ntours).^2 ./ (sumsq/ntours)
    TouringRunResults(results, xarray, trVs, totvis, rpacc, xplapac, toureff)    
end
