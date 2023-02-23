@testset "NRPTSampler.jl" begin
    # TODO: when a test for NRSTSampler exists, use the sampler built there
    @model function BetaBernoulli(obs)
        p ~ Beta(2,2)
        for i in eachindex(obs)
            obs[i] ~ Bernoulli(p)
        end
        p
    end
    tmbb = NRST.TuringTemperedModel(BetaBernoulli(bitrand(rng, 10)))
    ns, TE, Λ = NRSTSampler(
        tmbb,
        rng,
        use_mean=true,
        γ = 2.0,
        xpl_smooth_λ=1e-5,
        maxcor=0.9,
        verbose=false
    );
    nrpt = NRST.NRPTSampler(ns);
    tr   = NRST.run!(nrpt,rng,100);
    @testset "Communication" begin
        # check that the perm/sper identity holds after the run
        @test nrpt.perm[nrpt.sper] == collect(0:get_N(nrpt))
        # check that indices moved by at most one step at a time
        @test unique!(sort!(vec(mapslices(diff,tr.perms,dims=[2])))) == [-1,0,1]
    end
    @testset "Exploration" begin
        # check that taking one extra expl_step! aligns the explorers
        NRST.expl_step!(nrpt,rng)
        @test [ns.xpl.curβ[] for ns in nrpt.nss[nrpt.sper[2:end]]] == ns.np.betas[2:end]
        @test [NRST.params(ns.xpl) for ns in nrpt.nss[nrpt.sper[2:end]]] == ns.np.xplpars
    end
end
