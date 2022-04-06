@testset "Turing_interface.jl" begin
    # lognormal prior, no data
    @model function Lnormal()
        s ~ LogNormal()
    end
    lnormal = Lnormal()
    spl     = Sampler(HMC(0.1,5))
    rng     = Random.GLOBAL_RNG
    vi      = DynamicPPL.VarInfo(rng, lnormal)
    link!(vi, spl)
    @testset "gen_randref" begin
        # since randref produces transformed samples, they should be Normal()
        randref = gen_randref(lnormal, spl)
        thetas  = map(_ -> randref(rng)[1], 1:200)
        @test pvalue(OneSampleADTest(thetas, Normal())) > 0.05
    end
    @testset "gen_Vref" begin
        Vref = gen_Vref(vi,spl,lnormal)
        @test all(map(x-> Vref([x]) ≈ (-logpdf(Normal(), x)), randn(rng,10)))
    end
end

