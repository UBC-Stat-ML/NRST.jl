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
# compare self-tuning to truth
###############################################################################

true_c = F.(ns.np.betas) .- F(ns.np.betas[1]) # adjust to convention c(0)=0
NRST.tune!(samplers, verbose=true)
maximum(abs.(samplers[1].np.c[2:end] .- true_c[2:end]) ./ true_c[2:end]) < 0.1

###############################################################################
# check if using exact tuning gives uniform distribution
# I.e., set c(b) = F(b)
###############################################################################

copyto!(ns.np.c, F.(ns.np.betas))
results = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
variation(sum(results[:visits], dims=1)) < 0.05

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
    M = reduce(hcat,results[:xarray][i])'
    scatter!(p, M[:,1], M[:,2];x...)
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

# if d == 2
    # # plot!
    # xmax = 4*s0
    # xrange = -xmax:0.1:xmax
    # xlim = extrema(xrange)
    # minloglev = logpdf(MultivariateNormal(mu(0.), sbsq(0.)*I(d)), 3*s0*ones(2))
    # maxloglev = logpdf(MultivariateNormal(mu(1.), sbsq(1.)*I(d)), mu(1.))
    # lvls = exp.(range(minloglev,maxloglev,length=10))
    # plotvec = Vector(undef,length(ns.np.betas))
    # for (i,b) in enumerate(ns.np.betas)
    #     p = plot()
    #     draw_contour!(p,b,xrange)
    #     draw_points!(
    #         p,i;xlim=xlim,ylim=xlim,
    #         markeralpha=0.3,markerstrokewidth=0,
    #         legend_position=:bottomleft,label="b=$(round(b,digits=2))"
    #     )
    #     plotvec[i] = p
    # end
    # plot(plotvec..., size=(1600,800))
# end

###############################################################################
# check E^{b}[V] is accurately estimated
# compare F' to 
# - multithr touring with NRST : ✓
# - singlethr touring with NRST: ✓
# - serial sampling with NRST  : ✓
# - IID sampling from truth    : ✓
# - serial exploration kernels : ✓
# - parlel exploration kernels : ✓

aggV = similar(ns.np.c)
cvd_pal = :tol_bright
plot(F',0.,1., label="Theory", palette = cvd_pal) # ground truth

# parallel NRST
for (i, xs) in enumerate(results[:xarray])
    aggV[i] = mean(ns.np.V.(xs))
end
plot!(ns.np.betas, aggV, label="MT-Tour", palette = cvd_pal)

# iid sampling
meanv(b) = mean(ns.np.V.(eachcol(rand(MultivariateNormal(mu(b), sbsq(b)*I(d)),1000))))
plot!(ns.np.betas, meanv.(ns.np.betas), label="MC", palette = cvd_pal)

# use the explorers to approximate E^{b}[V]: single thread
for i in eachindex(aggV)
    if i==1
        aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
    else        
        traceV = similar(aggV, 5000)
        NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
        aggV[i] = mean(traceV)
    end
end
plot!(ns.np.betas, aggV, label="SerMH", palette = cvd_pal)

# use the explorers to approximate E^{b}[V]: multithread
Threads.@threads for i in eachindex(aggV)
    if i==1
        aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
    else        
        traceV = similar(aggV, 5000)
        NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
        aggV[i] = mean(traceV)
    end
end
plot!(ns.np.betas, aggV, label="ParMH", palette = cvd_pal)

# serial NRST: no tours involved
xtrace, iptrace = NRST.run!(ns, nsteps=10000)
aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
nvisits = zeros(Int, length(aggV))
for (n, ip) in enumerate(eachcol(iptrace))
    nvisits[ip[1] + 1] += 1
    aggV[ip[1] + 1]    += ns.np.V(xtrace[n])
end
aggV ./= nvisits
plot!(ns.np.betas, aggV, label="SerNRST", palette = cvd_pal)

# single thread NRST by tours
xtracefull = Vector{typeof(ns.x)}(undef, 0)
iptracefull = Matrix{Int}(undef, 2, 0)
ntours = 1000
for k in 1:ntours
    xtrace, iptrace = NRST.tour!(ns)
    append!(xtracefull, xtrace)
    iptracefull = hcat(iptracefull, reduce(hcat, iptrace))
end
aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
nvisits = zeros(Int, length(aggV))
for (n, ip) in enumerate(eachcol(iptracefull))
    nvisits[ip[1] + 1] += 1
    aggV[ip[1] + 1]    += ns.np.V(xtracefull[n])
end
aggV ./= nvisits
plot!(ns.np.betas, aggV, label="ST-Tour", palette = cvd_pal)
