##############################################################################
# various trace objects for SimulatedTempering samplers
# TODO: implement the "Recorders" approach, as in Pigeons. This gives much more
# flexibility to produce custom traces without needing to re-write code
##############################################################################

abstract type AbstractTrace{T,TI<:Int,TF<:AbstractFloat} end

function Base.similar(tr::TTrace) where {T,TI,TF,TTrace <: AbstractTrace{T,TI,TF}}
    TTrace.name.wrapper(T, get_N(tr), TF) # hack: https://discourse.julialang.org/t/get-generic-constructor-of-parametric-type/57189/2
end

#######################################
# ConstCostTrace: O(1) cost wrt tour length
#######################################

struct ConstCostTrace{T,TI<:Int,TF<:AbstractFloat} <: AbstractTrace{T,TI,TF}
    N::TI
    n_steps::Base.RefValue{TI}
    n_vis_top::Base.RefValue{TI}
    n_v_evals::Base.RefValue{TI}
end
get_N(tr::ConstCostTrace) = tr.N
get_nsteps(tr::ConstCostTrace) = tr.n_steps[]
get_nvtop(tr::ConstCostTrace) = tr.n_vis_top[]
get_nvevals(tr::ConstCostTrace) = tr.n_v_evals[]

function ConstCostTrace(::Type{T}, N::TI, ::Type{TF}, args...) where {T,TI<:Int,TF<:AbstractFloat}
    ConstCostTrace{T,TI,TF}(N, Ref(zero(TI)), Ref(zero(TI)), Ref(zero(TI)))
end

# trace postprocessing
function post_process(tr::ConstCostTrace, visacc::Matrix, args...)
    visacc[end, begin] += tr.n_vis_top[] # for simplicity, assign all visits to the top level to only one of the directions
    return
end

#######################################
# IPRPTrace: only ip and rejection probs
#######################################

struct IPRPTrace{T,TI<:Int,TF<:AbstractFloat} <: AbstractTrace{T,TI,TF}
    N::TI
    trIP::Vector{SVector{2,TI}} # can use a vector of SVectors since traces should not be modified
    trRP::Vector{TF}
end
get_N(tr::IPRPTrace) = tr.N
# get_nsteps and get_nvtop are defined below for Union{IPRPTrace,NRSTTrace}

function IPRPTrace(::Type{T}, N::TI, ::Type{TF}, args...) where {T,TI<:Int,TF<:AbstractFloat}
    IPRPTrace{T,TI,TF}(N, SVector{2,TI}[], TF[])
end

# trace postprocessing
function post_process(
    tr::IPRPTrace{T,I,K},
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rpacc::Matrix{K},          # size (N+1) × 2. accumulates rejection probs
    args...
    ) where {T,I,K}
    for (n, ip) in enumerate(tr.trIP)
        l      = first(ip)
        idx    = l + 1
        idxeps = (last(ip) == one(I) ? 1 : 2)
        visacc[idx, idxeps] += one(I)
        rpacc[ idx, idxeps] += tr.trRP[n]
    end
end

#######################################
# full trace
#######################################

struct NRSTTrace{T,TI<:Int,TF<:AbstractFloat} <: AbstractTrace{T,TI,TF}
    trX::Vector{T}
    trIP::Vector{SVector{2,TI}} # can use a vector of SVectors since traces should not be modified
    trV::Vector{TF}
    trRP::Vector{TF}
    trXplAP::Vector{Vector{TF}}
end

# outer constructors that allocate empty arrays
function NRSTTrace(::Type{T}, N::TI, ::Type{TF}) where {T,TI<:Int,TF<:AbstractFloat}
    trX     = T[]
    trIP    = SVector{2,TI}[]
    trV     = TF[]
    trRP    = TF[]
    trXplAP = [TF[] for _ in 1:N]
    NRSTTrace(trX, trIP, trV, trRP, trXplAP)
end
# outer constructors that allocate fixed size arrays
function NRSTTrace(::Type{T}, N::TI, ::Type{TF}, nsteps::Int) where {T,TI<:Int,TF<:AbstractFloat}
    trX     = Vector{T}(undef, nsteps)
    trIP    = Vector{SVector{2,TI}}(undef, nsteps)
    trV     = Vector{TF}(undef, nsteps)
    trRP    = similar(trV)
    trXplAP = [TF[] for _ in 1:N]                  # cant predict number of visits to each level
    NRSTTrace(trX, trIP, trV, trRP, trXplAP)
end
get_N(tr::NRSTTrace) = length(tr.trXplAP)  # recover N. cant do N=N(tr) because julia gets dizzy
get_nsteps(tr::Union{IPRPTrace,NRSTTrace}) = length(tr.trIP) # recover nsteps
function get_nvtop(tr::Union{IPRPTrace,NRSTTrace})
    N = get_N(tr)
    sum(ip -> first(ip)==N, tr.trIP)
end

function Base.show(io::IO, mime::MIME"text/plain", tr::NRSTTrace)
    println(io, "An NRSTTrace object with fields:")
    print(io, "X: ");show(io,mime,typeof(tr.trX))
    print(io, "\nIP:");show(io,mime,tr.trIP)
    print(io, "\nV:");show(io,mime,tr.trV)
    print(io, "\nRP:");show(io,mime,tr.trRP)
    print(io, "\nXPLAP:");show(io,mime,tr.trXplAP)
end

# trace postprocessing
function post_process(
    tr::NRSTTrace{T,I,K},
    visacc::Matrix{I},         # size (N+1) × 2. accumulates visits
    rpacc::Matrix{K},          # size (N+1) × 2. accumulates rejection probs
    xarray::Vector{Vector{T}}, # length = N+1. i-th entry has samples at state i
    trVs::Vector{Vector{K}},   # length = N+1. i-th entry has Vs corresponding to xarray[i]
    xplapac::Vector{K}         # length = N. accumulates explorers' acc probs
    ) where {T,I,K}
    i₀ = first(first(tr.trIP)) # first state in trace. for NRST, i_0==0, but not for others necessarily
    for (n, ip) in enumerate(tr.trIP)
        l      = first(ip)
        idx    = l + 1
        idxeps = (last(ip) == one(I) ? 1 : 2)
        visacc[idx, idxeps] += one(I)
        rpacc[ idx, idxeps] += tr.trRP[n]
        if l >= 1 && n > 1                                        # some STs (e.g. GT95) can renew at i_0=1, so no counting that one
            nvl = visacc[idx,1] + visacc[idx,2] - (l==i₀ ? 1 : 0) # (>=1) number of visits so far to level l, regardless of eps. since some STs can renew at i₀ neq 0, need to adjust accounting
            try
                xplapac[l] += tr.trXplAP[l][nvl]                
            catch e
                println("Error reading XplAP[l=$l][nvl=$nvl]: n=$n, i₀=$i₀.")
                rethrow(e)
            end
        end
        length(tr.trX) >= n && push!(xarray[idx], tr.trX[n])      # handle case keep_xs=false
        push!(trVs[idx], tr.trV[n])
    end
end

##############################################################################
# summaries of traces
##############################################################################

abstract type RunResults{T,TI<:Int,TF<:AbstractFloat} end

get_N(res::RunResults) = length(res.trVs)-1 # retrieve max tempering level
rejrates(res::RunResults) = res.rpacc ./ res.visits # matrix of rejection rates
averej(res::RunResults) = averej(rejrates(res))
averej(R::Matrix) = (R[1:(end-1),1] + R[2:end,2])/2

# estimate Λs using rejections rates estimated from a run
function get_lambdas(averej::Vector{K}) where {K<:AbstractFloat}
    pushfirst!(cumsum(averej), zero(K))
end

# tour effectiveness estimator under ELE assumption
toureffELE(ar::AbstractVector) = inv(1 + 2*sum(r->r/(1-r), ar)) # special case of symmetric rejections
toureffELE(R::AbstractMatrix)  = toureffELE(averej(R))          # TODO: implement general formula, instead of defaulting to symmetric case
toureffELE(res::RunResults)    = toureffELE(rejrates(res))

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

#######################################
# touring
#######################################

struct TouringRunResults{T,TI,TF,TTrace<:AbstractTrace{T,TI,TF}} <: RunResults{T,TI,TF}
    trvec::Vector{TTrace}     # vector of raw traces from each tour
    xarray::Vector{Vector{T}} # length = N+1. i-th entry has samples at level i
    trVs::Vector{Vector{TF}}  # length = N+1. i-th entry has Vs corresponding to xarray[i]
    visits::Matrix{TI}        # total number of visits to each (i,eps)
    rpacc::Matrix{TF}         # accumulates rejection probs of swaps started from each (i,eps)
    xplapac::Vector{TF}       # length N. accumulates explorers' acc probs
    toureff::Vector{TF}       # tour effectiveness for each i ∈ 0:N
end
get_ntours(res::TouringRunResults) = length(res.trvec)
tourlengths(res::TouringRunResults) = get_nsteps.(res.trvec)

# outer constructor that parses a vector of traces
function TouringRunResults(results::Vector{TST}) where {T,I,K,TST<:AbstractTrace{T,I,K}}
    ntours  = length(results)
    N       = get_N(first(results))
    xarray  = [T[] for _ in 0:N]                                   # i-th entry has samples at level i
    trVs    = [K[] for _ in 0:N]                                   # i-th entry has Vs corresponding to xarray[i]
    totvis  = zeros(I, N+1, 2)                                     # total visits to each (i,eps) state
    rpacc   = zeros(K, N+1, 2)                                     # accumulates rejection probs of swaps started from each (i,eps)
    xplapac = zeros(K, N)                                          # accumulates explorers' acc probs
    curvis  = Matrix{I}(undef, N+1, 2)                             # visits in current tour to each (i,eps) state
    sumsq   = zeros(K, N+1)                                        # accumulate (in float to avoid overflow) squared number of visits for each 0:N state (for tour effectiveness)

    # iterate tours
    for tr in results
        fill!(curvis, zero(I))                                     # reset tour visits
        try
            post_process(tr, curvis, rpacc, xarray, trVs, xplapac) # parse tour trace
        catch e
            println("Error processing a trace, dumping it:")
            display(tr); rethrow(e)
        end
        totvis .+= curvis                                          # accumulate total visits
        sumsq  .+= vec(sum(curvis, dims=2)).^2                     # squared number of visits to each of 0:N (regardless of direction)
    end
    
    # compute tour effectiveness and return
    toureff = vec(sum(totvis, dims=2).^2) ./ (ntours*sumsq)        # = (sum(totvis, dims=2)/ntours).^2 ./ (sumsq/ntours)
    map!(TE -> isnan(TE) ? zero(K) : TE, toureff, toureff)         # correction for unvisited levels
    TouringRunResults(
        results, xarray, trVs, totvis, rpacc, xplapac, toureff
    )    
end

