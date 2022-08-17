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
ns, ts = NRSTSampler(tm, rng, N=4);#;stage_1="iid");
nrpt = NRPTSampler(ns);
NRST.tune_explorers!(nrpt, rng, verbose=true);
tr  = run!(nrpt, rng, 128);
