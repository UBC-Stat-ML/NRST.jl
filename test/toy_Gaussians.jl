###############################################################################
# 
###############################################################################

using NRST
using Zygote
using StatsPlots, StatsBase
using Distributions
using LinearAlgebra

const d = 2
const s0 = 2.
const m = 4.
sbsq(b) = 1/(1/(s0*s0) + b)
mu(b) = b*m*sbsq(b)*ones(d)

ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x.-m*ones(d))), # likelihood: N(m1, I)
    x->(0.5sum(abs2,x)/(s0*s0)),    # reference: N(0, s0^2I)
    () -> s0*randn(d),              # reference: N(0, s0^2I)
    collect(range(0,1,length=9)),   # betas = uniform grid in [0,1]
    50,                             # nexpl
    false                           # tune using median energy
);

# tune and run
chan = NRST.tune!(ns, verbose=true);
results = NRST.parallel_run!(chan, ntours=400);

# plot contours of pdf of N(mu_b, s_b^2 I) versus scatter of samples
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

# plot!
xmax = 4*s0
xrange = -xmax:0.1:xmax
xlim = extrema(xrange)
minloglev = logpdf(MultivariateNormal(mu(0.), sbsq(0.)*I(d)), 3*s0*ones(2))
maxloglev = logpdf(MultivariateNormal(mu(1.), sbsq(1.)*I(d)), mu(1.))
lvls = exp.(range(minloglev,maxloglev,length=10))
plotvec = Vector(undef,length(ns.np.betas))
for (i,b) in enumerate(ns.np.betas)
    p = plot()
    draw_contour!(p,b,xrange)
    draw_points!(
        p,i;xlim=xlim,ylim=xlim,
        markeralpha=0.3,markerstrokewidth=0,
        legend_position=:bottomleft,label="b=$(round(b,digits=2))"
    )
    plotvec[i] = p
end
plot(plotvec..., size=(1600,800))

# true neg log partition == free energy
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end

# manual tuning
copyto!(ns.np.c, F.(ns.np.betas))
results = NRST.parallel_run!(chan, ntours=4000);
sum(results[:visits], dims=1) # not uniform...

# compare F' to NRST and to IID sampling from truth
# conclusion: F' and MC agree, NRST does not :(
plot(F',0.,1., label="Theory")
aggV = similar(ns.np.c)
for (i, xs) in enumerate(results[:xarray])
    aggV[i] = mean(ns.np.V.(xs))
end
plot!(ns.np.betas, aggV, label="NRST")
meanv(b) = mean(ns.np.V.(eachcol(rand(MultivariateNormal(mu(b), sbsq(b)*I(d)),1000))))
plot!(ns.np.betas, meanv.(ns.np.betas), label="MC")