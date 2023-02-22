###############################################################################
# exploration kernels
###############################################################################

abstract type ExplorationKernel end

# generic constructor that pre-computes common requirements
function (::Type{TXpl})(
    tm::TemperedModel,
    x, 
    β::AbstractFloat, 
    curV::Base.RefValue{<:AbstractFloat};
    kwargs...
    ) where {TXpl <: ExplorationKernel}
    vref = Vref(tm, x)
    vβ   = vref + β*curV[]
    TXpl(tm, x, Ref(β), Ref(vref), curV, Ref(vβ); kwargs...)
end

# copy constructor
function Base.copy(xpl::TXpl, newtm::TemperedModel, newx, ncurV::Base.RefValue{<:AbstractFloat}) where {TXpl <: ExplorationKernel}
    β    = xpl.curβ[]
    vref = Vref(newtm, newx)
    vβ   = vref + β*ncurV[]
    TXpl(xpl, newtm, newx, Ref(β), Ref(vref), ncurV, Ref(vβ))
end

# copy constructor
function Base.copy(xpl::ExplorationKernel)
    copy(xpl, copy(xpl.tm), copy(xpl.x), Ref(xpl.curV[]))
end

# by default, tuning does nothing
function tune!(::ExplorationKernel, args...; kwargs...) end

# same with smoothing
function smooth_params!(::Vector{<:ExplorationKernel}, args...) end

#######################################
# sampling methods
#######################################

# run sampler keeping track only of cummulative acceptance probability
# used in tuning
function run!(ex::ExplorationKernel, rng::AbstractRNG, nsteps::Int)
    sum_ap = zero(eltype(ex.curV))
    for _ in 1:nsteps
        ap, _   = step!(ex, rng)
        sum_ap += ap
    end
    return (sum_ap/nsteps)
end

# run sampler keeping track of V
function run!(ex::ExplorationKernel, rng::AbstractRNG, trV::Vector{K}) where {K}
    sum_ap = zero(K)
    nsteps = length(trV)
    for n in 1:nsteps
        ap, _   = step!(ex, rng)
        sum_ap += ap
        trV[n]  = ex.curV[]
    end
    return (sum_ap/nsteps)
end

#######################################
# methods interfacing with an NRSTSampler
#######################################

# explore for nsteps using the given params, without keeping track of anything
# note: this only makes sense when called by an NRSTSampler with which the
# explorer shares x and curV
function explore!(
    ex::ExplorationKernel,
    rng::AbstractRNG,
    β::AbstractFloat,
    params::NamedTuple,
    nsteps::Int
    )
    update_β!(ex, β)
    set_params!(ex, params)
    acc = zero(β)
    nvs = zero(nsteps)
    for _ in 1:nsteps
        ap, nv = step!(ex, rng)
        acc += ap
        nvs += nv
    end
    return acc/nsteps, nvs       # return average acceptance probability and number of V(x) evaluations
end
function update_β!(ex::ExplorationKernel, β::AbstractFloat)
    ex.curβ[]    = β
    ex.curVref[] = Vref(ex.tm, ex.x)          # vref is *not* shared so needs updating
    ex.curVβ[]   = ex.curVref[] + β*ex.curV[] # vβ is *not* shared so needs updating
end

default_nexpl_steps(ex::ExplorationKernel) = 10

#######################################
# methods for handling potentials
#######################################

# compute all potentials at some x
function potentials(ex::ExplorationKernel, newx)
    vref, v = potentials(ex.tm, newx)
    vβ = vref + ex.curβ[]*v
    vref, v, vβ
end

# return the potentials for the current point
potentials(ex::ExplorationKernel) = (ex.curVref[], ex.curV[], ex.curVβ[])

# set potentials. used during sampling with an explorer
function set_potentials!(ex::ExplorationKernel, vref::F, v::F, vβ::F) where {F<:AbstractFloat}
    ex.curVref[] = vref
    ex.curV[]    = v
    ex.curVβ[]   = vβ
end

#######################################
# methods for collections of exploration kernels
#######################################

# instantiate a vector of explorers by copying one. used for tuning NRSTSampler
# note: the resulting explorers are not tethered to any NRSTSampler, in the sense 
# that the state x is not shared with any NRSTSampler. Conversely, if xpl is
# the explorer of an NRSTSampler, this function does not change that.
function replicate(xpl::TXpl, betas::AbstractVector{<:AbstractFloat}) where {TXpl <: ExplorationKernel}
    N    = length(betas) - 1
    xpls = Vector{TXpl}(undef, N)
    for i in 1:N
        newxpl  = copy(xpl)           # use custom copy constructor
        update_β!(newxpl, betas[i+1]) # set β and recalculate Vβ
        xpls[i] = newxpl
    end
    return xpls
end