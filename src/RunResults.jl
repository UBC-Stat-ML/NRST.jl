#######################################
# raw trace of a serial run
#######################################

struct NRSTTrace{T,TI<:Int,TF<:AbstractFloat}
    trX::Vector{T}
    trIP::Vector{SVector{2,TI}}
    trV::Vector{TF}
    trRP::Vector{TF}
    trXplAP::Vector{Vector{TF}}
end

# outer constructor based on "examples" of types
function NRSTTrace(::Type{T}, N::TI,::Type{TF}) where {T,TI<:Int,TF<:AbstractFloat}
    trX     = T[]
    trIP    = SVector{2,TI}[]                # can use a vector of SVectors since traces should not be modified
    trV     = TF[]
    trRP    = TF[]
    trXplAP = [TF[] for _ in 1:N]
    NRSTTrace(trX, trIP, trV, trRP, trXplAP)
end
get_N(tr::NRSTTrace) = length(tr.trXplAP)  # recover N. cant do N=N(tr) because julia gets dizzy
get_nsteps(tr::NRSTTrace) = length(tr.trV) # recover nsteps
NRSTTrace(tr::NRSTTrace{T,TI,TF}) where {T,TI,TF} = NRSTTrace(T, get_N(tr), TF) # constructor empty trace of the same type that another
function Base.empty!(tr::NRSTTrace{T,TI,TF}) where {T,TI,TF}
    empty!(tr.trX)
    empty!(tr.trIP)
    empty!(tr.trV)
    empty!(tr.trRP)
    empty!.(tr.trXplAP) # empty the component vectors not the outer vector, so that it retains length N
end
function Base.resize!(tr::NRSTTrace{T,TI,TF}, n::Int) where {T,TI,TF}
    resize!(tr.trX, n)
    resize!(tr.trIP, n)
    resize!(tr.trV, n)
    resize!(tr.trRP, n)
end

#######################################
# trace postprocessing
#######################################

abstract type RunResults{T,TI<:Int,TF<:AbstractFloat} end

get_N(res::RunResults) = length(res.trVs)-1 # retrieve max tempering level

struct SerialRunResults{T,TI,TF} <: RunResults{T,TI,TF}
    tr::NRSTTrace{T,TI}       # raw trace
    xarray::Vector{Vector{T}} # i-th entry has samples at state i
    trVs::Vector{Vector{TF}}  # i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}        # total number of visits to each (i,eps)
    rpacc::Matrix{TF}         # accumulates rejection probs of swaps started from each (i,eps)
    xplapac::Vector{TF}       # length N. accumulates explorers' acc probs
end

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

