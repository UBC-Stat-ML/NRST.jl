using NRST
16*2^8
# test
# likelihood: N((1,1), I), reference: N(0, 4I)
# => -log-target = K + 1/2[(x-1)^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^T(x-1)-1^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^Tx - 21^Tx + 1/4x^Tx]
# = K + 1/2[(5/4)x^Tx - 2(5/4)(4/5)1^Tx]
# = K + 1/2(5/4)[x^Tx - 2(4/5)1^Tx]
# = K + 1/2(5/4)(x-4/5)^T(x-4/5)
# => target: N((4/5, 4/5), (4/5)I)

# 1/2[b(x-1)^T(x-1) + 1/4x^Tx]
# = 1/2[bx^T(x-1) - b1^T(x-1) + 1/4x^Tx]
# = 1/2[bx^Tx - 2b1^Tx + 1/4x^Tx] + 1/2[2b]
# = 1/2[(b+1/4)x^Tx - 2b(b+1/4)(b+1/4)^{-1}1^Tx] + b
# = 1/2(b+1/4)[x^Tx - 2[b/(b+1/4)]1^Tx] + b
# = 1/2(b+1/4)[x^Tx - 2[b/(b+1/4)]1^Tx + [b/(b+1/4)]1^T 1] + [b - b/(b+1/4)]
# = 1/2(b+1/4)(x-b/(b+1/4))^T(x-b/(b+1/4)) + b[1 - 1/(b+1/4)]
# = 1/2(b+1/4)(x-b/(b+1/4))^T(x-b/(b+1/4)) + b[(b-3/4)/(b+1/4)]
# => target(b): N(b/(b+1/4)(1, 1), (b+1/4)^{-1}I)
# => Z(b) = sqrt(2pi sigma^2(b))^2 exp(b[(b-3/4)/(b+1/4)])
# = 2pi(b+1/4)^{-1} exp(b[(b-3/4)/(b+1/4)])
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- [1.,1.])), # likelihood: N((1,1), I)
    x->(0.125sum(abs2,x)), # reference: N(0, 4I)
    () -> 2randn(2), # reference: N(0, 4I)
    collect(range(0,1,length=9)),
    50,
    false
);
ns.np.c
chan = NRST.tune!(ns,verbose=true)




# PROBLEM: tuning with mean is hard when there's a peak of E^{b}[V] around some b
# TODO: get analytic log(Z(b)) for problem above and tune c directly using that!
using StatsBase, StatsPlots
results = NRST.parallel_run!(chan,ntours=4000);
plot(vec(sum(results[:visits], dims=1)))
aggV = similar(ns.np.c)
for (i, xs) in enumerate(results[:xarray])
    aggV[i] = mean(winsor(ns.np.V.(xs), prop=0.05))
end
plot(aggV)
NRST.trpz_apprx!(ns.np.c, ns.np.betas, aggV)
plot(ns.np.c)
