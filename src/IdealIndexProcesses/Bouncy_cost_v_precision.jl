using DataFramesMeta, ColorSchemes, Gadfly, Distributions, Printf, DataFrames
import Cairo, Fontconfig # for saving as png

# set wd if working from REPL
isinteractive() && cd("/home/mbiron/Documents/RESEARCH/UBC_PhD/simulated_tempering/")

# load Bouncy definitions
# note: no need to tell include to look in code/jl folder. from the docs:
# "The included path, source.jl, is interpreted relative to the file where the include call occurs. This makes it simple to relocate a subtree of source files."
# https://docs.julialang.org/en/v1/manual/code-loading/
include("Bouncy.jl"); 

function sim_rep_tours(
    bouncy::Bouncy,
    K::Int, # run K iid tours
    nrep::Int # repeat simulations nrep times
    )
    times = Vector{Float64}(undef, K)
    counts = Vector{Int64}(undef, K)
    maxs = Vector{Float64}(undef, nrep)
    sums = Vector{Float64}(undef, nrep)
    for i in 1:nrep
        sim_tours!(bouncy,times,counts) # run K tours
        maxs[i] = maximum(times)
        sums[i] = sum(times)
    end
    maxs, sums
end

# # test
# sim_rep_tours(Bouncy(10),10,100)

function make_df(Lambda_vec,K_vec,nrep)
    df = DataFrame(K = Int64[], Lambda = Float64[], Grouping = String[], mean = Float64[], sdev = Float64[])
    for lam in Lambda_vec
        bouncy=Bouncy(lam)
        for K in K_vec
            # K=10
            maxs,sums=sim_rep_tours(bouncy,K,nrep)
            push!(df,[K, lam, "Parallel (max)", mean(maxs), std(maxs)])
            push!(df,[K, lam, "Serial (sum)", mean(sums), std(sums)])
        end
    end
    df
end

# plot cost versus accuracy digits (AD)
# The definition of AD is such that 
# AD_{n+1} = AD_{n}+1 whenever sd_{n+1} = sd_n/10
# Therefore, AD_{n} = C + log10(1/sd_n), since
# AD_{n+1} - AD_{n} = log10(1/sd_{n+1}) - log10(1/sd_n) = log10(sd_n/sd_{n+1}) = 1 
# But var = 1/(K TE) => sd = 1/sqrt(K TE), and therefore
# AD_n = C + 0.5(log10(K_n) + log10(TE))
# use empirical formula TE = 1/(1+2Λ) instead of estimate
# in particular, for K_n = 10^{2n}, we get
# AD_n = (C+ 0.5log10(TE)) + n --> as expected

# create plot and save
nrep = 200
df = make_df(10.0 .^(-3:3),round.(Int, 10.0 .^(0:0.5:5)),nrep)
rib_scale = 1/sqrt(nrep) # plot ribbon with ±1 std dev of the mean
# facet by Lambda
p = @linq df |>
    transform(
        :LambdaStr = map(y -> @sprintf("Λ=10<sup>%0.0f</sup>", y), log10.(:Lambda)),
        :AD = 0.5*log10.(:K ./ (1 .+ 2*(:Lambda))),
        :lo = :mean .- rib_scale*(:sdev),:hi = :mean .+ rib_scale*(:sdev)
    ) |>
    transform(:ADpos = :AD .- minimum(:AD)) |>
    rename(:Grouping => :Algorithm) |>
    plot(
        xgroup = :LambdaStr, x = :ADpos, y = :mean, color = :Algorithm,
        ymin = :lo, ymax = :hi, alpha=[0.4],
        Geom.subplot_grid(Geom.line, Geom.ribbon),
        Scale.y_log10,
        Guide.xlabel("Digits of accuracy"), Guide.ylabel("Time to completion"),
        Theme(background_color=color("white"))
)
draw(PNG("./plots/time_to_complete_acc_digits.png", 18inch, 4inch), p)
# # facet by algorithm
# p = @linq df |>
#     transform(
#         :AD = 0.5*log10.(:K ./ (1 .+ 2*(:Lambda))),
#         :lo = :mean .- rib_scale*(:sdev),:hi = :mean .+ rib_scale*(:sdev)
#     ) |>
#     transform(:ADpos = :AD .- minimum(:AD)) |>
#     plot(
#         xgroup = :Grouping, x = :ADpos, y = :mean, color = :Lambda,
#         ymin = :lo, ymax = :hi, alpha=[0.4],
#         Geom.subplot_grid(Geom.line, Geom.ribbon,free_y_axis=true),
#         # Scale.x_log10, 
#         Scale.y_log10,
#         Scale.color_discrete(n -> get(ColorSchemes.viridis, range(0, 1, length=n))),
#         Guide.xlabel("Digits of accuracy"), Guide.ylabel("Time to completion"),
#     )
