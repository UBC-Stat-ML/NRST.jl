@testset "SliceSampler.jl" begin
    @model function ToyModel()
        y1 ~ Normal(7,0.1)
        y2 ~ Laplace(-3,5)
        y3 ~ Gumbel()
    end
    tm  = NRST.TuringTemperedModel(ToyModel())
    rng = SplittableRandom(1)
    x   = rand(tm,rng)
    v   = NRST.V(tm,x)
    nxs = 10000

    @testset "SliceSamplerStepping" begin 
        ss  = NRST.SliceSamplerSteppingOut(tm, x, 1.0, Ref(v));
        xs  = collect(hcat(map(_ -> (NRST.step!(ss,rng); copy(ss.x)),1:nxs)...)');
        @test all(
            [
                pvalue(
                    ApproximateOneSampleKSTest(
                        xs[:,i], 
                        first(getproperty(tm.viout.metadata,sym).dists) 
                    )
                ) 
            for (i,sym) in enumerate(propertynames(tm.viout.metadata))
            ] .> 0.01
        )
    end

    @testset "SliceSamplerDoubling" begin 
        ss  = NRST.SliceSamplerDoubling(tm, x, 1.0, Ref(v));
        xs  = collect(hcat(map(_ -> (NRST.step!(ss,rng); copy(ss.x)),1:nxs)...)');
        @test all(
            [
                pvalue(
                    ApproximateOneSampleKSTest(
                        xs[:,i], 
                        first(getproperty(tm.viout.metadata,sym).dists) 
                    )
                ) 
            for (i,sym) in enumerate(propertynames(tm.viout.metadata))
            ] .> 0.01
        )
    end
end
