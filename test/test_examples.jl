using NRST
using DynamicPPL, Distributions

# lognormal prior, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x .~ Normal(0.,s)
end
model = Lnmodel(randn(30))
tm    = NRST.TuringTemperedModel(model);
rng   = SplittableRandom(4)
ns, ts = NRSTSampler(tm, rng, N=4, tune=false);
nrpt = NRPTSampler(ns);
NRST.tune_explorers!(nrpt, rng, verbose=true);
ns.np.betas
ns.np.c
res      = @timed NRST.tune_c_betas!(ns.np, nrpt, rng, 32);
ns.np.betas
ns.np.c
Δβs,Λ,ar = res.value # note: average rejections are before grid adjustment, so they are technically stale, but are still useful to assess convergence. compute std dev of average of up and down rejs
ar_ratio = std(ar)/mean(ar)
