###############################################################################
# Toy example in ℝᵈ with
# - Prior     : N(0, s_0^2 I)
# - Likelihood: N(m1, I)
# Implies Gaussian annealing path: b ↦ N(mu_b, s_b^2 I), with
#     s_b^2 := (s_0^{-2} + b)^{-1},     mu_b := b m s_b^2 1
# Using un-normalized Vref and V (i.e., using V = negative of what is inside
# the exp() in the Gaussian pdf), we get that the free energy is given by
#     F(b) := -log(Z(b)) = -0.5d(log(2pi s_b^2) - bm^2[1 - bs_b^2])
# Using the thermodynamic identity, we can obtain the function b ↦ E^{b}[V] by
# differentiating F (with Zygote)
###############################################################################

using NRST
using Zygote
using StatsBase
using StatsPlots
using Distributions
using LinearAlgebra

const d    = 2 # contour plots only for d==2 
const s0   = 2.
const m    = 4.
const s0sq = s0*s0
sbsq(b)    = 1/(1/s0sq + b)
mu(b)      = b*m*sbsq(b)*ones(d)

# true free energy function == -log(Z(b))
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end

# build an NRST sampler, tune exploration kernels, and do initial tuning of c
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- m)),     # likelihood: N(m1, I)
    x->(0.5sum(abs2,x)/s0sq),     # reference: N(0, s0^2I)
    () -> s0*randn(d),            # reference: N(0, s0^2I)
    collect(range(0,1,length=9)), # betas = uniform grid in [0,1]
    50,                           # nexpl
    true                          # tune c using mean energy
);

# build vector of identical copies of ns for safe parallel computations
samplers = NRST.copy_sampler(ns, nthrds = Threads.nthreads());

###############################################################################
# compare energy and free-energy estimates to truth
###############################################################################

NRST.tune!(samplers, verbose=true)
restune = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
meanV   = [mean(ns.np.V.(xs)) for xs in restune[:xarray]]

# energy
p1 = plot(F',0.,1., label="Theory", title="Mean energy")
plot!(p1, ns.np.betas, meanV, label="Estimate", seriestype=:scatter)

# free-energy
p2 = plot(
    b -> F(b)-F(0.), 0., 1., label="Theory", legend_position = :bottomright,
    title = "Free energy"
)
plot!(p2, ns.np.betas, samplers[1].np.c, label="Estimate", seriestype = :scatter)
plot(p1,p2,size=(800,400))

###############################################################################
# check if we achieve uniform distribution over levels
# compare exact c v. tuned c
###############################################################################

copyto!(ns.np.c, F.(ns.np.betas))
resexact = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
xs = repeat(1:length(ns.np.c), 2)
gs = repeat(["Exact c", "Tuned c"], inner=length(ns.np.c))
ys = vcat(vec(sum(resexact[:visits], dims=1)),vec(sum(restune[:visits], dims=1)))
groupedbar(
    xs,ys,group=gs, legend_position=:topleft,
    title="Number of visits to every level"
)

###############################################################################
# plot contours of pdf of N(mu_b, s_b^2 I) versus scatter of samples
###############################################################################

function draw_contour!(p,b,xrange)
    dist = MultivariateNormal(mu(b), sbsq(b)*I(d))
    f(x1,x2) = pdf(dist,[x1,x2])
    Z = f.(xrange, xrange')
    contour!(p,xrange,xrange,Z,levels=lvls,aspect_ratio = 1)
end

function draw_points!(p,i;x...)
    M = reduce(hcat,restune[:xarray][i])
    scatter!(p, M[1,:], M[2,:];x...)
end

if d == 2
    # plot!
    xmax = 4*s0
    xrange = -xmax:0.1:xmax
    xlim = extrema(xrange)
    minloglev = logpdf(MultivariateNormal(mu(0.), sbsq(0.)*I(d)), 3*s0*ones(2))
    maxloglev = logpdf(MultivariateNormal(mu(1.), sbsq(1.)*I(d)), mu(1.))
    lvls = exp.(range(minloglev,maxloglev,length=10))
    anim = @animate for (i,b) in enumerate(ns.np.betas)
        p = plot()
        draw_contour!(p,b,xrange)
        draw_points!(
            p,i;xlim=xlim,ylim=xlim,
            markeralpha=0.3,markerstrokewidth=0,
            legend_position=:bottomleft,label="β=$(round(b,digits=2))"
        )
        plot(p)
    end
    gif(anim, fps = 2)
end

###############################################################################
# check scaling of max(nsteps) and max(times) as ntours → ∞
# use exact c
# run twice to avoid including compilation times
###############################################################################

function max_scaling(samplers, nrounds, nreps)
    ntours  = Threads.nthreads()*round.(Int, 2 .^(0:(nrounds-1)))
    msteps  = Matrix{Int}(undef, nreps, nrounds)
    mtimes  = Matrix{Float64}(undef, nreps, nrounds)
    for rep in 1:nreps
        for r in 1:nrounds
            res = NRST.parallel_run!(samplers, ntours=ntours[r])
            msteps[rep,r] = maximum(res[:nsteps])
            mtimes[rep,r] = maximum(res[:times])
        end
    end
    return ntours, msteps, mtimes
end

ntours, msteps, mtimes = max_scaling(samplers, 2, 2)
ntours, msteps, mtimes = max_scaling(samplers, 10, 50)
p1 = plot()
p2 = plot()
for (r,m) in enumerate(eachrow(msteps))
    plot!(p1, ntours, m, xaxis=:log, linealpha = 0.2, label="", linecolor = :blue)
    plot!(p2, ntours, mtimes[r,:], xaxis=:log, linealpha = 0.2, label="", linecolor = :blue)
end
plot!(
    p1, ntours, vec(sum(msteps,dims=1))/size(msteps,1), xaxis=:log,
    linewidth = 2, label="Average", linecolor = :blue, legend_position=:topleft,
    xlabel="Number of tours", ylabel="Maximum number of steps across tours"
)
plot!(
    p2, ntours, vec(sum(mtimes,dims=1))/size(mtimes,1), xaxis=:log,
    linewidth = 2, label="Average", linecolor = :blue, legend_position=:topleft,
    xlabel="Number of tours", ylabel="Time to complete longest tour (s)"
)
using Plots.PlotMeasures
plot(p1,p2,size=(1000,450), margin = 4mm)

###############################################################################
# compute TE and ESS (use mcmcse.jl and traditional) as difficulty increases. For example: 
# - m → ∞
# - d → ∞
# PROBLEM: these parameters are set as const
###############################################################################


# ###############################################################################
# # check E^{b}[V] is accurately estimated
# # compare F' to 
# # - multithr touring with NRST : ✓
# # - singlethr touring with NRST: ✓
# # - serial sampling with NRST  : ✓
# # - IID sampling from truth    : ✓
# # - serial exploration kernels : ✓
# # - parlel exploration kernels : ✓
# ###############################################################################

# cvd_pal = :tol_bright
# plot(F',0.,1., label="Theory", palette = cvd_pal) # ground truth

# # parallel NRST
# aggV = similar(ns.np.c)
# for (i, xs) in enumerate(resexact[:xarray])
#     aggV[i] = mean(ns.np.V.(xs))
# end
# plot!(ns.np.betas, aggV, label="MT-Tour", palette = cvd_pal)

# # iid sampling
# meanv(b) = mean(ns.np.V.(eachcol(rand(MultivariateNormal(mu(b), sbsq(b)*I(d)),1000))))
# plot!(ns.np.betas, meanv.(ns.np.betas), label="MC", palette = cvd_pal)

# # use the explorers to approximate E^{b}[V]: single thread
# for i in eachindex(aggV)
#     if i==1
#         aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
#     else        
#         traceV = similar(aggV, 5000)
#         NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
#         aggV[i] = mean(traceV)
#     end
# end
# plot!(ns.np.betas, aggV, label="SerMH", palette = cvd_pal)

# # use the explorers to approximate E^{b}[V]: multithread
# Threads.@threads for i in eachindex(aggV)
#     if i==1
#         aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
#     else        
#         traceV = similar(aggV, 5000)
#         NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
#         aggV[i] = mean(traceV)
#     end
# end
# plot!(ns.np.betas, aggV, label="ParMH", palette = cvd_pal)

# # serial NRST: no tours involved
# xtrace, iptrace = NRST.run!(ns, nsteps=10000)
# aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
# nvisits = zeros(Int, length(aggV))
# for (n, ip) in enumerate(eachcol(iptrace))
#     nvisits[ip[1] + 1] += 1
#     aggV[ip[1] + 1]    += ns.np.V(xtrace[n])
# end
# aggV ./= nvisits
# plot!(ns.np.betas, aggV, label="SerNRST", palette = cvd_pal)

# # single thread NRST by tours
# xtracefull = Vector{typeof(ns.x)}(undef, 0)
# iptracefull = Matrix{Int}(undef, 2, 0)
# ntours = 1000
# for k in 1:ntours
#     xtrace, iptrace = NRST.tour!(ns)
#     append!(xtracefull, xtrace)
#     iptracefull = hcat(iptracefull, reduce(hcat, iptrace))
# end
# aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
# nvisits = zeros(Int, length(aggV))
# for (n, ip) in enumerate(eachcol(iptracefull))
#     nvisits[ip[1] + 1] += 1
#     aggV[ip[1] + 1]    += ns.np.V(xtracefull[n])
# end
# aggV ./= nvisits
# plot!(ns.np.betas, aggV, label="ST-Tour", palette = cvd_pal)

