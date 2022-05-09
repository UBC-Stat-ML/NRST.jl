using StatsPlots, Distributions, Printf, BenchmarkTools
using DataFrames, GLM

# set wd if working from REPL
isinteractive() && cd("/home/mbiron/Documents/RESEARCH/UBC_PhD/simulated_tempering/")

# load Bouncy definitions
# note: no need to tell include to look in code/jl folder. from the docs:
# "The included path, source.jl, is interpreted relative to the file where the include call occurs. This makes it simple to relocate a subtree of source files."
# https://docs.julialang.org/en/v1/manual/code-loading/
include("Bouncy.jl"); 

function visits2top_stats(
    Lambda_vec=10.0 .^(-3:3), # vector of Lambda values to use in simulations
    K=1000000 # number of iid tours used to estimate moments
    )
    # allocate storage
    times = Vector{Float64}(undef, K)
    counts = Vector{Int64}(undef, K)
    means = Vector{Float64}(undef, length(Lambda_vec))
    sdevs = Vector{Float64}(undef, length(Lambda_vec))

    # iterate Lambdas
    for (i,Lambda) in enumerate(Lambda_vec)
        #i=4;Lambda = Lambda_vec[4]
        sim_tours!(Bouncy(Lambda), times, counts)
        means[i] = mean(counts)
        sdevs[i] = std(counts)
    end

    # plot means with ±2 std. dev. ribbon
    plot_data = DataFrame(Lambda=Lambda_vec,mean=means, sdev=sdevs)
    rib_scale = 2/sqrt(K) # plot ribbon with ±2 std dev of the mean
    means_plot = @df plot_data plot(
        :Lambda,:mean,ribbon=rib_scale*sdevs, ylims=(0.8,1.2),
        xscale = :log10, xlabel="Λ", ylabel="Mean number of visits to top",
        label=""# label=@sprintf("Λ = %.1f",lam), legend = :topleft
    );
    # plot Variance vs Lambda, add ols line
    vars_plot = @df plot_data plot(
        xscale=:log10,yscale=:log10,
        :Lambda,:sdev .^2,seriestype = :scatter, label="Data", legend = :topleft,
        xlabel="Λ", ylabel="Variance of number of visits to top"
    );
    ols = lm(@formula(log(sdev^2) ~ log(Lambda)),plot_data)
    slope=round(exp(coef(ols)[1]);digits=1); pow=round(coef(ols)[2];digits=1)
    logL = log10.(extrema(Lambda_vec))
    plot!( # add trend
        vars_plot,
        10.0 .^(logL[1]:0.1:logL[2]),
        x -> slope*(x^pow),
        linestyle = :dash,
        label=@sprintf("var ≈ %.1f*Λ^(%.1f) (ols)",slope,pow)
    );
    # plot TE vs Lambda, add curve implied from the one for variance
    plot_data[!,:TE] = 1 ./ (1 .+ (sdevs ./ means).^2) # compute TE
    TE_plot = @df plot_data plot(
        xscale=:log10,
        :Lambda,:TE,seriestype = :scatter, label="Data",
        xlabel="Λ", ylabel="Tour Effectiveness"
    );
    plot!( # add trend
        TE_plot,
        10.0 .^(logL[1]:0.1:logL[2]),
        x -> 1 / (1 + slope*(x^pow)),
        linestyle = :dash,
        label=@sprintf("TE ≈ 1/(1+%.1f*Λ^(%.1f)) (ols)",slope,pow)
    );
    # arrange in 1x3 layout
    display(plot(
        means_plot,vars_plot,TE_plot,layout=(1,3), size=(1350,450),
        plot_title=@sprintf("Tour statistics using %d iid tours per Λ value",K),
        left_margin = 7Plots.mm, bottom_margin = 7Plots.mm
    ))
end

# plot and save
visits2top_stats()
savefig("./plots/tour_effectiveness.png")