@testset "Turing_interface.jl" begin
    # lognormal prior, normal likelihood
    @model function Lnmodel(x)
        s ~ LogNormal()
        x .~ Normal(0.,s)
    end
    model = Lnmodel(randn(30))
    tm    = NRST.TuringTemperedModel(model)
    ls    = [randn()]
    @testset "Vref" begin
        # check that Vref matches the -logpdf of the lognormal plus logabsdetjac
        @test abs(NRST.Vref(tm, ls) - (-logpdf(LogNormal(), exp(first(ls)))- first(ls))) < 1e-12
    end
    @testset "V" begin
        # check V matches the likelihood and does not include logabsdetjac
        @test abs(NRST.V(tm, ls) - mapreduce(x->-logpdf(Normal(0.,exp(first(ls))),x), +, model.args.x)) < 1e-12
    end
end

