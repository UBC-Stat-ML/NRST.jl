#########################################################################################
# interfacing with the Turing.jl environment
# TODO:
# - use DynamicPPL @models, evaluate V_b(x) := Vref(x) + bV(x) with MiniBatchContext()
#########################################################################################

using Random, Distributions, AbstractMCMC, AdvancedHMC, DynamicPPL, Turing, HypothesisTests

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x ~ Normal(0.,s)
end
lnmodel = Lnmodel(-.1234)
rng = Random.GLOBAL_RNG

# check logposterior is accurately computed
vi = VarInfo(rng, lnmodel, SampleFromPrior())
s = vi[@varname(s)]
getlogp(vi) == logpdf(LogNormal(),s) + logpdf(Normal(0.,s),lnmodel.args[1])

####################################################
# below I'm loosely following DynamicPPL.initialstep
####################################################

!istrans(vi,@varname(s))           # variables are currently in constrained space (s>0)
spl = Sampler(HMC(0.1,5), lnmodel) # create basic HMC sampler for the model
link!(vi, spl)                     # convert the variables to unconstrained form
istrans(vi,@varname(s))

# this call to evaluate!! implicitly uses the DefaultContext, which does not
# sample, only calculates logp. And since now the variables are transformed,
# logp should account for the transformation logdetjac
vi = last(DynamicPPL.evaluate!!(lnmodel, rng, vi, spl))
theta = vi[spl]
theta[1] == log(s) # DynamicPPL.Bijectors.bijector(LogNormal())() == Bijectors.Log{0}()
getlogp(vi) == (logpdf(LogNormal(),s) + log(s)) + logpdf(Normal(0.,s),lnmodel.args[1])

# create a Hamiltonian
metricT = Turing.Inference.getmetricT(spl.alg)
metric = metricT(length(theta))
∂logπ∂θ = Turing.Inference.gen_∂logπ∂θ(vi, spl, lnmodel)
logπ = Turing.Inference.gen_logπ(vi, spl, lnmodel)
logπ(theta) == getlogp(vi)
hamiltonian = AdvancedHMC.Hamiltonian(metric, logπ, ∂logπ∂θ)
z = AdvancedHMC.phasepoint(rng, theta, hamiltonian)
log_density_old = getlogp(vi)
ϵ = spl.alg.ϵ
kernel = Turing.Inference.make_ahmc_kernel(spl.alg, ϵ)
t = AdvancedHMC.transition(rng, hamiltonian, kernel, z)
adaptor = Turing.Inference.AHMCAdaptor(spl.alg, hamiltonian.metric; ϵ=ϵ) # no adaptation because HMC(0.1,5) is not adaptive
if spl.alg isa AdaptiveHamiltonian
    # nadapts is passed to check that i<=nadapts, otherwise nothing is done: https://github.com/TuringLang/AdvancedHMC.jl/blob/7cad9f02e6eb4095ad8b2e7ef06d3474dfb19135/src/sampler.jl#L80
    hamiltonian, kernel, _ =
        AdvancedHMC.adapt!(hamiltonian, kernel, adaptor,
                    1, nadapts, t.z.θ, t.stat.acceptance_rate)
end
if t.stat.is_accept
    vi = DynamicPPL.setindex!!(vi, t.z.θ, spl)
    vi = DynamicPPL.setlogp!!(vi, t.stat.log_density)
else
    vi = DynamicPPL.setindex!!(vi, theta, spl)
    vi = DynamicPPL.setlogp!!(vi, log_density_old)
end
transition = Turing.Inference.HMCTransition(vi, t)
transition.θ
transition.lp
transition.stat
state = Turing.Inference.HMCState(vi, 1, kernel, hamiltonian, t.z, adaptor)

# loop
sample, state = Turing.AbstractMCMC.step(rng, lnmodel, spl, state)
state.i

####################################################
# now do this using the step methods
####################################################

# initial step
_, state = DynamicPPL.AbstractMCMC.step(rng, lnmodel, spl)
s = state.vi[@varname(s)]
state.vi[spl][1] == log(s)
getlogp(state.vi) == state.hamiltonian.ℓπ(state.vi[spl])

# subsequent steps
_, state = Turing.AbstractMCMC.step(rng, lnmodel, spl, state)

####################################################
# evaluating tempered posteriors
####################################################

β = 0.7
vi = VarInfo(rng, lnmodel, SampleFromPrior())
s = vi[@varname(s)]
getlogp(vi) == logpdf(LogNormal(),s) + logpdf(Normal(0.,s),lnmodel.args[1])
mbctx = MiniBatchContext(DefaultContext(),β)
vi = last(DynamicPPL.evaluate!!(lnmodel, vi, mbctx))
getlogp(vi) == logpdf(LogNormal(),s) + β*logpdf(Normal(0.,s),lnmodel.args[1])

spl = Sampler(HMC(0.1,5), lnmodel) # create basic HMC sampler for the model
link!(vi, spl)                    # convert the variables to unconstrained form
istrans(vi,@varname(s))
vi = last(DynamicPPL.evaluate!!(lnmodel, rng, vi, spl, mbctx)) # since mbctx uses DefaultContext, this does not sample, only evaluates
theta = vi[spl]
theta[1] == log(s)
getlogp(vi) == (logpdf(LogNormal(),s) + log(s)) + β*logpdf(Normal(0.,s),lnmodel.args[1])

####################################################
# sampling from the prior
####################################################

function gen_randref(model::Model, spl::Sampler)
    function randref(rng)
        vi = VarInfo(rng, model, SampleFromPrior())
        link!(vi, spl)
        vi[spl]
    end
    return randref
end
randref = gen_randref(lnmodel, spl)
thetas = map(_ -> randref(rng)[1], 1:100)
pvalue(OneSampleADTest(thetas, Normal())) > 0.05

#########################################################################################
# TODO:
# - use AdvancedHMC samplers as exploration kernels. Cannot really use the Turing because
#   the model for the explorer is only specified through the tempered energy V_b, which is
#   more similar to the standard AdvancedHMC interface. however, must be careful about link/unlik
#   to handle transformations 
#
# Outline of how Turing adapts AdvancedHMC to work with trans variables, within AdvancedMCMC framework
# 1) the sampling starts with a call to "sample"
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L103
#    - after setting default params, this call is redirected to the "mcmcsample" method
#	https://github.com/TuringLang/AbstractMCMC.jl/blob/3de7393b8b8e76330f53505b27d2b928ef178681/src/sample.jl#L90
# 2) the 1st "step" call is done without a "state" object
#	https://github.com/TuringLang/AbstractMCMC.jl/blob/3de7393b8b8e76330f53505b27d2b928ef178681/src/sample.jl#L120
#    For HMC, this call without state is captured here
# 	https://github.com/TuringLang/DynamicPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L71
#    - if no initial parameters are passed, it uses the "initialsampler" method of the sampler to 
#      draw an initial value. Turing defaults to "SampleFromUniform", as shown here
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L100
#    - NOTE: currently "SampleFromUniform" is replaced in DynamicPPL by "SamplerFromPrior". see
# 	https://github.com/TuringLang/DynamicPPL.jl/blob/5cc158556a8742194647cf72ae3e3cad2718fac2/src/sampler.jl#L97
#      These samples from there prior occur in the constrained (untransformed) space. 
#    - after setting initial value in constrained space, the "initialstep" method is called
#	https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L143
#      essentially, the function 
#         a) transforms the initial value and compute the logposterior
#         b) creates a Hamiltonian object, which has a metric, a fn to eval the logdens, and a fn
#	     to eval its gradient
#	  c) does a transtion 
#	  d) does adaptation
#	  e) creates a new HMCState, used for the next call 
