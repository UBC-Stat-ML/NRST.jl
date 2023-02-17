@testset "SliceSampler.jl" begin
    @model function ToyModel()
        y ~ Normal()
    end
    tm  = NRST.TuringTemperedModel(ToyModel())
    rng = SplittableRandom()
    x   = [randn(rng)]
    ps  = NRST.potentials(tm,x)
    ss  = NRST.SliceSampler(
        tm, x, Ref(1.0), Ref(ps[1]), Ref(0.0), Ref(ps[1])
    );
    nsteps=30000
    xs = similar(x, nsteps)
    for i in 1:nsteps
        NRST.step!(ss,rng)
        xs[i] = ss.x[1]
    end
    @test pvalue(OneSampleADTest(xs, Normal())) > 0.1
end
