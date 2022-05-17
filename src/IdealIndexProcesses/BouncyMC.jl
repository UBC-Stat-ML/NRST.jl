###############################################################################
# regenerative simulation of the non-reversible + equirejection Markov chain
# on 0:N
###############################################################################

struct BouncyMC{TF<:AbstractFloat, TI<:Int, TM<:AbstractMatrix{TF}} <: Bouncy
    R::TM                    # (N+1)×2 matrix of rejection probs. R[:,1] is up, R[:,2] is dn, R[1,2]=R[N+1,1]=1.0
    N::TI                    # levels are 0:N
    state::MVector{3,TI}     # current state (time, position, direction)
    nhits::Base.RefValue{TI} # number of hits above
end
function BouncyMC(R::AbstractMatrix{<:AbstractFloat})
    N = size(R,1)-1
    BouncyMC(R, N, MVector{3,typeof(N)}(undef), Ref(zero(N)))
end
BouncyMC(ρ::AbstractFloat, N::Int) = BouncyMC(Fill(ρ, (N+1, 2)))

# force a renewal on a BouncyMC + reset counter
function renew!(bouncy::BouncyMC{TF,TI}) where {TF,TI}
    bouncy.state[1] = zero(TI)
    bouncy.state[2] = zero(TI)
    bouncy.state[3] = one(TI)
    bouncy.nhits[]  = zero(TI)
end

# simulate a tour: starting at (0,+) until absorption into (0,-)
# i(-1) = -a + b = 2
# i(1)  =  a + b = 1
# => b=3/2 , a=-1/2
# =? i(e) = (3-e)÷2
function tour!(bouncy::BouncyMC{TF,TI};verbose::Bool=false) where {TF,TI}
    renew!(bouncy)
    N       = bouncy.N
    t, y, e = bouncy.state            # read state
    while true
        t     += one(TI)
        verbose && println((t,y,e))
        next_y = y + e                # attempt move
        if next_y > N                 # bounce above
            y = one(TI)
            e = -one(TI)
            bouncy.nhits[] += one(TI) # increment bounce-above counter
        elseif next_y < zero(TI)      # bounce below and exit loop
            y = zero(TI)
            e = one(TI)
            break
        elseif rand() < bouncy.R[y+one(TI), e>zero(TI) ? 1 : 2]
            e = -e
        else
            y = next_y
        end
    end
    # save last state into bouncy object
    bouncy.state[1] = t
    bouncy.state[2] = y
    bouncy.state[3] = e
    return
end

# get iid tours, record tour length and number of hits at the top
function run_tours!(
    bouncy::BouncyMC{TF,TI},
    times::Vector{TI},
    counts::Vector{TI};
    kwargs...
    ) where {TF,TI}
    ntours = length(times)
    for k in 1:ntours
        tour!(bouncy;kwargs...)
        times[k]  = bouncy.state[1] # tour length
        counts[k] = bouncy.nhits[]  # number of bounces at the top within the tour
    end
end
function run_tours!(bouncy::BouncyMC{TF,TI}, ntours::Int;kwargs...) where {TF,TI}
    times  = Vector{TI}(undef, ntours)
    counts = Vector{TI}(undef, ntours)
    run_tours!(bouncy, times, counts;kwargs...)
    return (times=times, counts=counts)
end
