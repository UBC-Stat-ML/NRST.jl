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
ns.curV[]
nrpt = NRPTSampler(ns);
tr = run!(nrpt, rng, 10000);
tr.rpsum/(tr.n[]/2)