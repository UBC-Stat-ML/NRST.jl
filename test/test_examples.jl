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
# [ns.curV[] for ns in nrpt.nss]
NRST.step!(nrpt, rng)
