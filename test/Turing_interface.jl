# TODO: update this to new Turing interface
# @testset "Turing_interface.jl" begin
#     # lognormal prior, normal likelihood
#     @model function Lnmodel(x)
#         s ~ LogNormal()
#         x .~ Normal(0.,s)
#     end
#     rng     = Random.GLOBAL_RNG
#     xobs    = randn(rng, 30) #-.1234
#     lnormal = Lnmodel(xobs)
#     spl     = Sampler(HMC(0.1,5)) # only used to enforce (0,∞) → ℝ trans via `link!`
#     vi      = DynamicPPL.VarInfo(rng, lnormal)
#     link!(vi, spl)
#     @testset "gen_randref" begin
#         # since randref produces transformed samples, they should be Normal()
#         randref = gen_randref(lnormal, spl)
#         thetas  = map(_ -> randref(rng)[1], 1:400)
#         @test pvalue(OneSampleADTest(thetas, Normal())) > 0.05
#     end
#     @testset "gen_Vref" begin
#         # check that Vref matches the -logpdf of log(LogNormal()) == Normal()
#         Vref = gen_Vref(vi,spl,lnormal)
#         @test all(map(θ-> Vref([θ]) ≈ -logpdf(Normal(), θ), randn(rng,10)))
#     end
#     @testset "gen_V" begin
#         V = gen_V(vi,spl,lnormal)
#         @test all(map(θ-> V([θ])+θ ≈ -sum(logpdf.(Normal(0.,exp(θ)),xobs)), randn(rng,10))) # "+θ" accounts for logdetjac, which apparently is passed in the LikelihoodContext, when it really should be passed with the prior
#     end
# end

