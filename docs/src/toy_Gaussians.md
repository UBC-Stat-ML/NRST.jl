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


### Efficacy of tuning

Here we run the tuning routine on our collection of samplers and compare the resulting `c` vector to the analytical expression for $\mathcal{F}(\cdot)$.
```@example tg; continued = false
NRST.tune!(samplers, verbose=true)
```
Let us compute the maximum absolute relative deviation with respect to the truth
```@example tg
true_c = F.(ns.np.betas) .- F(ns.np.betas[1]) # adjust to convention c(0)=0
maximum(abs.(samplers[1].np.c[2:end] .- true_c[2:end]) ./ true_c[2:end])
```

### Uniformity under exact tuning

Likewise, we can tune the vector `c` using the analytic expression for the free energy
```@example tg
copyto!(ns.np.c, F.(ns.np.betas))
```

!!! note "`np` is shared"
    The `np` field is shared across `samplers`, so by running the line above we are effectively changing the setting for all of them.

As mentioned in [Analytical form of the partition function](@ref), we would expect to obtain a uniform distribution over levels. Let us confirm this by running 512 tours per thread in parallel
```@example tg
results = NRST.parallel_run!(samplers, ntours=512*Threads.nthreads());
```
The number of visits to each state within each tour is stored in the field `results[:visits]`. Summing over all tours gives
```@example tg
sum(results[:visits], dims=1)
```
which is indeed fairly uniform.


### Visual inspection of samples

Here we compare the contours of the pdf of the annealed distributions versus the samples at each of the levels. First we write a function to add a contour to a plot
```@example tg
function draw_contour!(p,b,xrange)
    dist = MultivariateNormal(mu(b), sbsq(b)*I(d))
    f(x1,x2) = pdf(dist,[x1,x2])
    Z = f.(xrange, xrange')
    contour!(p,xrange,xrange,Z,levels=lvls,aspect_ratio = 1)
end
```
Next we write a function to add scatter plots of samples. Note that these are collected in the field `results[:xarray]`, which is a vector of length $N+1$. Its `i`-th entry contains a vector of samples from `i`-th annealed distribution, of length equal to the total number of visits to that level.
```@example tg
function draw_points!(p,i;x...)
    M = reduce(hcat,results[:xarray][i])'
    scatter!(p, M[:,1], M[:,2];x...)
end
```
Finally, we use these two functions to produce an animation that loops over the annealing levels
```@example tg
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


### Estimating the average energy

In section [Efficacy of tuning](@ref) we assessed the correctness of the approximation of
```math
\mathcal{F}(b)=-\log(\mathcal{Z}(\beta)) = \int_0^\beta \mathrm{d}\ b \mathbb{E}^{(b)}[V]
```
Instead, in this section we focus directly on $\mathbb{E}^{(b)}[V]$. The exact value for these quantities can be obtained by differentiating $\mathcal{F}(b)$
```math
\frac{\mathrm{d}}{\mathrm{d}b}\mathcal{F}(b)= \mathbb{E}^{(b)}[V]
```
Here we use the "prime operator" `'` from [Zygote](https://fluxml.ai/Zygote.jl/) to carry out this calculation through automatic differentation.
```@example tg; continued = true
p = plot(F',0.,1., label="Theory", title="Expected value of the energy");
```
We contrast this with the approximations obtained through NRST
```@example tg; continued = false
aggV = similar(ns.np.c)
for (i, xs) in enumerate(results[:xarray])
    aggV[i] = mean(ns.np.V.(xs))
end
plot!(p,ns.np.betas, aggV, label="NRST", seriestype=:scatter)
plot(p)
```
