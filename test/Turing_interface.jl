@testset "Turing_interface.jl" begin
    @testset "normal-lognormal" begin 
        # lognormal prior, normal likelihood
        @model function Lnmodel(x)
            s ~ LogNormal()
            x .~ Normal(0.,s)
        end
        model = Lnmodel(randn(30))
        tm    = NRST.TuringTemperedModel(model)
        ls    = [randn()]
        # check that Vref matches the -logpdf of the lognormal plus logabsdetjac
        @test abs(NRST.Vref(tm, ls) - (-logpdf(LogNormal(), exp(first(ls)))- first(ls))) < 1e-12
        # check V matches the likelihood and does not include logabsdetjac
        @test abs(NRST.V(tm, ls) - mapreduce(x->-logpdf(Normal(0.,exp(first(ls))),x), +, model.args.x)) < 1e-12
    end

    @testset "Dirichlet prior" begin
        @model function DirichletModel()
            x ~ Dirichlet(5, 1.0)
        end
        tm = NRST.TuringTemperedModel(DirichletModel());
        x  = rand(tm,Random.default_rng())
        @test !(0.0 ≈ NRST.Vref(tm,x))                  # although prior is uniform, we should be getting the logabsdetjac here
        @test 0.0 ≈ NRST.V(tm,x)                        # no observation -> 0 loglikelihood
    end
end

