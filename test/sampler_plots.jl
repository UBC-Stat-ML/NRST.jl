using NRST
using Plots

# create sampler
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- [1.,1.])), # likelihood: N((1,1), I)
    x->(0.125sum(abs2,x)), # reference: N(0, 4I)
    () -> 2randn(2), # reference: N(0, 4I)
    collect(range(0,1,length=9)),
    50
);

# plot index process
xtrace,iptrace = NRST.run!(ns,100)
plot(iptrace[1,:],label="",seriestype = :steppost)

