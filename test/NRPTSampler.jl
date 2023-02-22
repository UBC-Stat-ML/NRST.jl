@testset "NRPTSampler.jl" begin
    # TODO: when a test for NRSTSampler exists, use the sampler built there
    @model function BetaBernoulli(obs)
        p ~ Beta(2,2)
        for i = 1:length(obs)
            obs[i] ~ Bernoulli(p)
        end
        p
    end
    rng  = SplittableRandom()
    tmbb = NRST.TuringTemperedModel(BetaBernoulli(bitrand(10)))
    ns, TE, Λ = NRSTSampler(
        tmbb,
        rng,
        use_mean=true,
        γ = 2.0,
        xpl_smooth_λ=1e-5,
        maxcor=0.9,
    );
    nrpt = NRST.NRPTSampler(ns);
    tr   = NRST.run!(nrpt,rng,100);
    @testset "Communication" begin
        # check that indices move by at most one step at a time
        @test unique!(sort!(vec(mapslices(diff,tr.perms,dims=[2])))) == [-1,0,1]
    end
end
