using Plots, Distributions, Printf, BenchmarkTools

# set wd if working from REPL
isinteractive() && cd("/home/mbiron/Documents/RESEARCH/UBC_PhD/simulated_tempering/")

# load Bouncy definitions
# note: no need to tell include to look in code/jl folder. from the docs:
# "The included path, source.jl, is interpreted relative to the file where the include call occurs. This makes it simple to relocate a subtree of source files."
# https://docs.julialang.org/en/v1/manual/code-loading/
include("Bouncy.jl"); 

# get iid samples of max(iid tour times)
function sim_max_tours(
    bouncy::Bouncy,
    gr_size::Int, # take max of gr_size copies of abs time
    n_groups::Int # return n_groups simulations of the max
    )
    times = Vector{Float64}(undef, gr_size)
    counts = Vector{Int64}(undef, gr_size)
    maxs = Vector{Float64}(undef, n_groups)
    for i in 1:n_groups
        sim_tours!(bouncy,times,counts)
        maxs[i] = maximum(times)
    end
    maxs
end

# # test
# max_vec = sim_max_tours(Bouncy(2),50,1000)
# @btime sim_max_tours(Bouncy(2),50,1000) # O(1) allocations
# histogram(max_vec)
# extrema(max_vec)

# visualization
# plot average of the max versus gr_size, for fixed n_groups
function vis_max_growth(Lambda_vec,gr_size_vec,n_groups)
    rib_scale = 2/sqrt(n_groups) # plot ribbon with ±2 std dev of the mean
    plot_list = []
    for lam in Lambda_vec
        bouncy=Bouncy(lam)
        max_stats = map(
            g -> (v=sim_max_tours(bouncy,g,n_groups);[mean(v),std(v)]),
            gr_size_vec
        )
        max_stats = reduce(hcat, max_stats)'
        push!(plot_list, plot(
            gr_size_vec,max_stats[:,1],ribbon=rib_scale*max_stats[:,2],
            xscale = :log10, label=@sprintf("Λ = %.1f",lam), legend = :topleft
        ))
    end
    display(plot(plot_list...))
end

# create plot and save
vis_max_growth(10.0 .^(0:3),round.(Int, 10.0 .^(0:0.5:5)),200)
savefig("./plots/max_trend.png")