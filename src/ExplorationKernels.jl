###############################################################################
# exploration kernels
###############################################################################

abstract type ExplorationKernel end

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

#######################################
# methods used only within an explorer
#######################################

# compute all potentials at some x
function potentials(ex::ExplorationKernel, newx)
    vref, v = potentials(ex.tm, newx)
    vβ = vref + ex.curβ[]*v
    vref, v, vβ
end

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