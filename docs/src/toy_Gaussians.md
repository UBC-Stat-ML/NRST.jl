```@meta
CurrentModule = NRST
```

# A toy example involving Gaussians

Consider the following statistical model for $x\in\mathbb{R}^d$
```math
\begin{aligned}
x   &\sim \mathcal{N}_d(0, s_0^2 I) \\
y|x &\sim \mathcal{N}_d(x, I).
\end{aligned}
```
Suppose that we observe $y = (m,\dots,m) = m1$ for some $m \in \mathbb{R}$. Then, a potential function $V$ compatible with this observation is
```math
V(x) := \frac{1}{2}(x - m1)^\top(x - m1).
```
For the prior, we use an unnormalized potential $V_\text{ref}$ of the form
```math
V_\text{ref}(x) := \frac{1}{2s_0^2}x^\top x.
```

## Analytical form of the partition function

For $b\in[0,1]$, the annealed potential $V_b$ is given by
```math
\begin{aligned}
V_b(x) &:= V_0(x) + bV(x)\\
&= \frac{1}{2}\left[ s_0^{-2}x^\top x  + b(x-m1)^\top (x-m1) \right] \\
&= \frac{1}{2}\left[ s_0^{-2}x^\top x  + bx^\top x -2bm 1^\top x +bm^21^\top 1 \right] \\
&= \frac{1}{2}\left[ (s_0^{-2}+b)x^\top x  -2bm 1^\top x  \right] + \frac{1}{2}bdm^2 \\
&= \frac{1}{2}(s_0^{-2}+b)\left[x^\top x  -2[bm(s_0^{-2}+b)^{-1}1]^\top x  \right] + \frac{1}{2}bdm^2 \\
&= \frac{1}{2}(s_0^{-2}+b)\left[x^\top x  -2[bm(s_0^{-2}+b)^{-1}1]^\top x  + [b^2m^2(s_0^{-2}+b)^{-2}]1^\top 1 \right] \\
&\phantom{=} + \frac{1}{2}bdm^2 -  \frac{1}{2}(s_0^{-2}+b)[b^2m^2(s_0^{-2}+b)^{-2}]d \\
&= \frac{1}{2s_b^2}(x - \mu_b)^\top (x - \mu_b) + \frac{bdm^2}{2}\left[1 - bs_b^2\right],
\end{aligned}
```
where
```math
\mu_b := bms_b^21, \qquad s_b^2 := (s_0^{-2}+b)^{-1}.
```
Therefore, $V_b$ corresponds to the energy of a $\mathcal{N}_d(\mu_b, s_b^2I )$ distribution. From this we may infer the normalizing constant $\mathcal{Z}(b)$ associated to every $V_b$
```math
\begin{aligned}
\mathcal{Z}(b) &:= \int \mathrm{d} x \exp\left(-V_b(x)\right) \\
&= \exp\left(-\frac{bdm^2}{2}\left[1 - bs_b^2\right]\right) \int \mathrm{d} x \exp\left(-\frac{1}{2s_b^2}(x - \mu_b)^\top (x - \mu_b)\right) \\
&= [2\pi s_b^2]^{d/2} \exp\left(-\frac{bdm^2}{2}\left[1 - bs_b^2\right]\right).
\end{aligned}
```
It follows that the free energy $\mathcal{F}$ function is
```math
\begin{aligned}
\mathcal{F}(b) &= -\log(\mathcal{Z}(b)) \\
&= -\frac{d}{2}\log(2\pi s_b^2) +\frac{bdm^2}{2}\left[1 - bs_b^2\right] \\
&= -\frac{d}{2}\left(\log(2\pi s_b^2) -bm^2\left[1 - bs_b^2\right] \right).
\end{aligned}
```

!!! note "Exact tuning"
    As we know from theory, by using $c(b) = \mathcal{F}(b)$ we should obtain a uniform distribution over the indices.


## Experiments

We begin by importing necessary packages
```@example tg
using NRST
using Zygote
using StatsBase
using StatsPlots
using Distributions
using LinearAlgebra
```
Define the parameters of the problem
```@example tg; continued = true
const d    = 2
const s0   = 2.
const m    = 4.
const s0sq = s0*s0;
```
Using these we can write expressions for $\mu_b$, $s_b^2$, and $\mathcal{F}$
```@example tg; continued = true
sbsq(b) = 1/(1/s0sq + b)
mu(b)   = b*m*sbsq(b)*ones(d)
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end
```
The following statement initializes an `NRSTSampler` object with the energy functions of the problem. It also carries out the tuning of exploration kernels as well as a basic initial tuning of the `c` function.
```@example tg; continued = true
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- m)),     # likelihood: N(m1, I)
    x->(0.5sum(abs2,x)/s0sq),     # reference: N(0, s0^2I)
    () -> s0*randn(d),            # reference: N(0, s0^2I)
    collect(range(0,1,length=9)), # betas = uniform grid in [0,1]
    50,                           # nexpl
    true                          # tune c using mean energy
);
```
Since we wish to run NRST in parallel, we construct `nthreads()` identical copies of `ns` so that each thread can work independently of the others, thus avoiding race conditions.
```@example tg; continued = true
samplers = NRST.copy_sampler(ns, nthrds = Threads.nthreads());
```
The original `ns` object is stored in the first entry `samplers[1]`.


### Assessing estimates of mean energy and free energy 

We want to understand how well our sampler approximates $\mathbb{E}^{(\beta)}[V]$ and $\mathcal{F}(\beta)$. To do this, we first carry out the tuning procedure. The free energy estimates correspond to `c` under the current tuning strategy. For the mean energy, we run the sampler again, and using the resulting `ParallelRunResults` object, we request an estimation that also returns an approximation of the asymptotic Monte Carlo variance
```@example tg; continued = true
NRST.tune!(samplers, verbose=true)
restune = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
means, vars = NRST.estimate(restune, ns.np.V)
```
Let us visually inspect the results. We use the estimated variance to produce an approximate 95% confidence interval around each point estimate.
```@example tg; continued = false
# energy
p1 = plot(F',0.,1., label="Theory", title="Mean energy")
plot!(
    p1, ns.np.betas, means, label="Estimate", seriestype=:scatter,
    yerror = 1.96sqrt.(vars/restune.ntours)
)

# free-energy
p2 = plot(
    b -> F(b)-F(0.), 0., 1., label="Theory", legend_position = :bottomright,
    title = "Free energy"
)
plot!(p2, ns.np.betas, samplers[1].np.c, label="Estimate", seriestype = :scatter)

plot(p1,p2,size=(800,400))
```
The first figure shows a high level of agreement between the estimated mean energies and the theoretical values. The second plot shows more disagreement betweeen the theoretical and approximated values of the free energy. This means that the discrepancy could be corrected by having a finer grid.


### Uniformity of the distribution over levels

Under the mean-energy tuning strategy, we expect a uniform distribution over the levels. We can check this by inspecting the trace of the previous section. Also, we can do the same for the case where we set `c` to the exact value of the free energy
```@example tg; continued = true
copyto!(ns.np.c, F.(ns.np.betas))
resexact = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
NRST.full_postprocessing!(resexact) # computes the :visits field and others
```

!!! note "`np` is shared"
    The `np` field is shared across `samplers`, and `ns=samplers[1]`. Thus, by changing `ns.np.c` we are effectively changing the setting for all the samplers.

Let us visually inspect the results
```@example tg; continued = false
xs = repeat(1:length(ns.np.c), 2)
gs = repeat(["Exact c", "Tuned c"], inner=length(ns.np.c))
ys = vcat(vec(sum(resexact.visits, dims=1)),vec(sum(restune.visits, dims=1)))
groupedbar(
    xs,ys,group=gs, legend_position=:topleft,
    title="Number of visits to every level"
)
```
Setting `c` to its theoretical value under the mean-energy strategy indeed gives a very uniform distribution over levels. The case where `c` is tuned, on the other, has a slight bias towards the levels closer to the target measure.

### Scaling of the maximal tour length and duration

Here the samplers are run multiple rounds, using an exponentially increasing number of tours. For each round, we compute the maximal tour length and duration in seconds. Each round is repeated `nreps` times to assess the variability of these measurements.
```@example tg; continued = true
function max_scaling(samplers, nrounds, nreps)
    ntours  = Threads.nthreads()*round.(Int, 2 .^(0:(nrounds-1)))
    msteps  = Matrix{Int}(undef, nreps, nrounds)
    mtimes  = Matrix{Float64}(undef, nreps, nrounds)
    for rep in 1:nreps
        for r in 1:nrounds
            res = NRST.parallel_run!(samplers, ntours=ntours[r])
            NRST.tour_durations!(res) # populates only the :nsteps and :times fields
            msteps[rep,r] = maximum(res.nsteps)
            mtimes[rep,r] = maximum(res.times)
        end
    end
    return ntours, msteps, mtimes
end
```
We must run the function first to avoid counting compilation times
```@example tg; continued = true
ntours, msteps, mtimes = max_scaling(samplers, 2, 2)
```
Now we may proceed with the experiment
```@example tg; continued = false
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
```


### Visual inspection of samples

Here we compare the contours of the pdf of the annealed distributions versus the samples obtained at each of the levels when NRST is run using the tuned `c`. As we know from the theory, ergodicity holds for any (reasonable) `c`, so we should expect to see agreement between contours and samples.

```@example tg
function draw_contour!(p,b,xrange)
    dist = MultivariateNormal(mu(b), sbsq(b)*I(d))
    f(x1,x2) = pdf(dist,[x1,x2])
    Z = f.(xrange, xrange')
    contour!(p,xrange,xrange,Z,levels=lvls,aspect_ratio = 1)
end

function draw_points!(p,i;x...)
    M = reduce(hcat,restune.xarray[i])
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
```
We see that the samples correctly describe the pdf of the corresponding distributions.

