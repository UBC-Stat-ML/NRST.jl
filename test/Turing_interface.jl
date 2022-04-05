@testset "Turing_interface.jl" begin
    # lognormal prior, no data
    @model function Lnormal()
        s ~ LogNormal()
    end
    lnormal = Lnormal()
    spl     = Sampler(HMC(0.1,5), lnormal)
    @testset "gen_randref" begin
        # since randref produces transformed samples, they should be Normal()
        randref = gen_randref(lnormal, spl)
        thetas  = map(_ -> randref()[1], 1:100)
        @test pvalue(OneSampleADTest(thetas, Normal())) > 0.05
    end
end

