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
```@example tg
const d    = 2
const s0   = 2.
const m    = 4.
const s0sq = s0*s0;
```
Using these we can write expressions for $\mu_b$, $s_b^2$, and $\mathcal{F}$
```@example tg
sbsq(b) = 1/(1/s0sq + b)
mu(b)   = b*m*sbsq(b)*ones(d)
function F(b)
    ssq = sbsq(b)
    -0.5*d*( log(2*pi*ssq) - b*m*m*(1-b*ssq) )
end
```
The following statement initializes an `NRSTSampler` object with the energy functions of the problem. It also carries out the tuning of exploration kernels as well as a basic initial tuning of the `c` function.
```@example tg
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- m)),     # likelihood: N(m1, I)
    x->(0.5sum(abs2,x)/s0sq),     # reference: N(0, s0^2I)
    () -> s0*randn(d),            # reference: N(0, s0^2I)
    collect(range(0,1,length=9)), # betas = uniform grid in [0,1]
    50,                           # nexpl
    true                          # tune c using mean energy
);
```
```@repl
Threads.nthreads()
```