using NRST

# test
# likelihood: N((1,1), I), reference: N(0, 4I)
# => -log-target = K + 1/2[(x-1)^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^T(x-1)-1^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^Tx - 21^Tx + 1/4x^Tx]
# = K + 1/2[(5/4)x^Tx - 2(5/4)(4/5)1^Tx]
# = K + 1/2(5/4)[x^Tx - 2(4/5)1^Tx]
# = K + 1/2(5/4)(x-4/5)^T(x-4/5)
# => target: N((4/5, 4/5), (4/5)I)
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- [1.,1.])), # likelihood: N((1,1), I)
    x->(0.125sum(abs2,x)), # reference: N(0, 4I)
    () -> 2randn(2), # reference: N(0, 4I)
    collect(range(0,1,length=9)),
    50,
    false
);
results = NRST.tour!(ns);
results[1][2]
results = NRST.parallel_run!(ns,ntours=4000);
sum(results[2][:visits] .> 0)

using Plots
plot(results[2][:ip][:,1])