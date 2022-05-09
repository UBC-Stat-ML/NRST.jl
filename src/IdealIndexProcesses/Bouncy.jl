###############################################################################
# simulation of the simple PDMP with exp(Λ) times and reflecting boundaries
# in [0,1] 
###############################################################################

struct Bouncy{TF<:AbstractFloat,TI<:Int}
    Λ::TF                 # rate parameter
    dist::Exponential{TF} # jump distribution
    state::Vector{TF}     # current state (time, position, direction)
    counters::Vector{TI}  # keeps stats of events
end

function Bouncy(Λ::AbstractFloat)
    @assert Λ>0
    Bouncy(
        Λ,
        Exponential(1.0/Λ),          # Distributions uses the reversed convention
        Vector{typeof(Λ)}(undef, 3),
        zeros(Int,1)                 # 1 counter = number of bounces at the top
    )
end

# # test
# bouncy = Bouncy(15)
# isapprox(mean(rand(bouncy.dist,1000)), 1/bouncy.Λ; rtol=0.1)

# force a renewal on a Bouncy + reset counters
function renew!(bouncy::Bouncy{TF,TI}) where {TF,TI}
    bouncy.state[1] = zero(TF)
    bouncy.state[2] = zero(TF)
    bouncy.state[3] = one(TF)
    fill!(bouncy. counters, zero(TI))
end

# simulate a tour: starting at (0,+) until absorption into (0,-)
function tour!(bouncy::Bouncy{TF,TI}) where {TF,TI}
    renew!(bouncy)
    t, y, e = bouncy.state # read state
    # trace = isa(Y, Array) # check if user passed valid storage for trace
    # (trace) && push!(Y,[t, y, e]) # initialize storage if requested
    while true
        tau = rand(bouncy.dist)     # sample a jump time
        next_y = y + e * tau        # attempt move at unit speed in e direction
        if next_y>1                 # bounce above
            t = t+1-y
            y = one(TF)
            e = -one(TF)
            bouncy.counters[1] += 1 # increment bounce-above counter
        elseif next_y<0             # bounce below and exit loop
            t = t + y
            y = zero(TF)
            e = one(TF)
            break
        else                        # accept move
            t = t + tau
            y = next_y
            e = -e
        end
        # (trace) && push!(Y,[t, y, e]) # store data if requested
    end
    # save last state into bouncy object
    bouncy.state[1] = t
    bouncy.state[2] = y
    bouncy.state[3] = e
    # (trace) && push!(Y,[t, y, e]) # store last point if requested
    return
end

# # test
# bouncy = Bouncy(1000)
# @btime renew!(bouncy); tour!(bouncy) # 0 allocations
# bouncy.state[2:3] == [0.0,1.0]
# # plot trace
# Y = []
# tour!(Bouncy(10),Y)
# M = reduce(hcat,Y)'
# plot(M[:,1],M[:,2],xlim=(0.0,4.0),ylim=(0.0,1.0))

# get iid tours, record tour length and number of hits at the top
function run_tours!(
    bouncy::Bouncy{TF,TI},
    times::Vector{TF},
    counts::Vector{TI}
    ) where {TF,TI}
    ntours = length(times)
    for k in 1:ntours
        tour!(bouncy)
        times[k]  = bouncy.state[1]    # tour length
        counts[k] = bouncy.counters[1] # number of bounces at the top within the tour
    end
end
function run_tours!(bouncy::Bouncy{TF,TI}, ntours::Int) where {TF,TI}
    times  = Vector{TF}(undef, ntours)
    counts = Vector{TI}(undef, ntours)
    run_tours!(bouncy, times, counts)
    return (times=times, counts=counts)
end

# # test
# bouncy = Bouncy(100)
# K = 100000
# times = Vector{Float64}(undef, K)
# counts = Vector{Int64}(undef, K)
# # @btime run_tours!(bouncy,times,counts) # 0 allocations
# run_tours!(Bouncy(0.00100),times, counts)
# mean(counts) # constant with Λ
# std(counts) # increases with Λ

toureff(nhits::Vector{<:Int}) = inv(1 + (std(nhits) / mean(nhits))^2)
