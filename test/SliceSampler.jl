@testset "SliceSampler.jl" begin
    @model function ToyModel()
        y1 ~ Normal(7,0.1)
        y2 ~ Laplace(-3,5)
        y3 ~ Gumbel()
    end
    toytm = NRST.TuringTemperedModel(ToyModel())
    x     = rand(toytm,rng)
    v     = NRST.V(toytm,x)
    nxs   = 10000

    @testset "SliceSamplerStepping" begin 
        ss  = NRST.SliceSamplerSteppingOut(toytm, x, 1.0, Ref(v));
        xs  = collect(hcat(map(_ -> (NRST.step!(ss,rng); copy(ss.x)),1:nxs)...)');
        @test all(
            [
                pvalue(
                    ApproximateOneSampleKSTest(
                        xs[:,i], 
                        first(getproperty(toytm.viout.metadata,sym).dists) 
                    )
                ) 
            for (i,sym) in enumerate(propertynames(toytm.viout.metadata))
            ] .> 0.01
        )
    end

    @testset "SliceSamplerDoubling" begin 
        ss  = NRST.SliceSamplerDoubling(toytm, x, 1.0, Ref(v));
        xs  = collect(hcat(map(_ -> (NRST.step!(ss,rng); copy(ss.x)),1:nxs)...)');
        @test all(
            [
                pvalue(
                    ApproximateOneSampleKSTest(
                        xs[:,i], 
                        first(getproperty(toytm.viout.metadata,sym).dists) 
                    )
                ) 
            for (i,sym) in enumerate(propertynames(toytm.viout.metadata))
            ] .> 0.01
        )
    end
end
