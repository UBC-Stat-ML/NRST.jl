using NRST

# create sampler
ns=NRST.NRSTSampler(
    x->(0.5sum(abs2,x .- [1.,1.])), # likelihood: N((1,1), I)
    x->(0.125sum(abs2,x)), # reference: N(0, 4I)
    () -> 2randn(2), # reference: N(0, 4I)
    collect(range(0,1,length=9)),
    50
);

# create another sampler using the same NRSTProblem
ns2 = NRST.NRSTSampler(ns.np,ns.nexpl)

# note that ns.np and ns2.np now point to the same object
ns.np.betas
ns2.np.betas
ns.np.betas[1] = -1.2
ns2.np.betas[5] = 5.2
ns.np.betas[5]
