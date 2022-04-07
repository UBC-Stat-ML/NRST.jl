#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Random, Distributions, DynamicPPL, Turing

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x .~ Normal(0.,s)
end
lnmodel = Lnmodel(randn(30)) # -.1234
rng = Random.GLOBAL_RNG

# check logposterior is accurately computed
vi = VarInfo(rng, lnmodel)
s = vi[@varname(s)]
getlogp(vi) ≈ logpdf(LogNormal(),s) + sum(logpdf.(Normal(0.,s),lnmodel.args[1]))
tonamedtuple(vi)
####################################################
# below I'm loosely following DynamicPPL.initialstep
####################################################

!istrans(vi,@varname(s))  # variables are currently in constrained space (s>0)
spl = Sampler(HMC(0.1,5)) # create basic HMC sampler for the model
link!(vi, spl)            # convert the variables to unconstrained form
istrans(vi,@varname(s))
s == vi[@varname(s)]      # the getindex method for (vi,varname) does the conversion using Bijectors.invlink: https://github.com/TuringLang/DynamicPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L911

s == tonamedtuple(vi)[:s][1][1] # 

# this call to evaluate!! implicitly uses the DefaultContext, which does not
# sample, only calculates logp. And since now the variables are transformed,
# logp should account for the transformation logdetjac
vi = last(DynamicPPL.evaluate!!(lnmodel, rng, vi, spl))
theta = vi[spl]
theta[1] == log(s) # DynamicPPL.Bijectors.bijector(LogNormal())() == Bijectors.Log{0}()
getlogp(vi) ≈ (logpdf(LogNormal(),s) + log(s)) + sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

####################################################
# step with HMC
####################################################

# initial step
t, state = DynamicPPL.AbstractMCMC.step(rng, lnmodel, spl)
Turing.Inference.getparams(t)
Turing.Inference.getparams(state.vi)
Turing.Inference.metadata(t)

s = state.vi[@varname(s)]
state.vi[spl][1] == log(s)
getlogp(state.vi) == state.hamiltonian.ℓπ(state.vi[spl])

# subsequent steps
_, state = Turing.AbstractMCMC.step(rng, lnmodel, spl, state)

####################################################
# evaluating tempered posteriors
# does NOT work when link!'d
####################################################

β = 0.7
vi = VarInfo(rng, lnmodel)
s = vi[@varname(s)]
getlogp(vi) ≈ logpdf(LogNormal(),s) + sum(logpdf.(Normal(0.,s),lnmodel.args[1]))
mbctx = MiniBatchContext(DefaultContext(),β)
vi = last(DynamicPPL.evaluate!!(lnmodel, vi, mbctx))
getlogp(vi) ≈ logpdf(LogNormal(),s) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

link!(vi, spl)
istrans(vi,@varname(s))
vi = last(DynamicPPL.evaluate!!(lnmodel, rng, vi, spl, mbctx)) # since mbctx uses DefaultContext, this does not sample, only evaluates
theta = vi[spl]
theta[1] == log(s)
getlogp(vi) ≈ (logpdf(LogNormal(),s) + log(s)) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

using NRST

# this works :)
V    = gen_V(vi, spl, lnmodel)
Vref = gen_Vref(vi, spl, lnmodel)
-(Vref(theta) + β*V(theta)) ≈ (logpdf(LogNormal(),s) + log(s)) + β*sum(logpdf.(Normal(0.,s),lnmodel.args[1]))

####################################################
# inference with NRST using the Turing interface
####################################################

using Random, Distributions, DynamicPPL, NRST, Plots, StatsBase

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x .~ Normal(0.,s)
end
lnmodel = Lnmodel(randn(30)) # -.1234
betas = range(0,1,15) .^(2) #pushfirst!(10. .^range(-2,0,9), 0.)
ns=NRSTSampler(
    lnmodel, # use the Turing model to build potentials and prior sampling
    betas,
    50,      # nexpl
    true     # tune c using mean energy
);
xtrace, iptrace = NRST.run!(ns,nsteps=10000)
print(sort(collect(countmap(iptrace[1,:])), by=x->x[1]))
plot(iptrace[1,:])

# inspect samples at the top chain
idx = iptrace[1,:] .== (length(ns.np.betas)-1)
ss = exp.(vcat(xtrace[idx]...)) # transform to constrained space
plot(ss)
histogram(ss)

# run in parallel
[get_num_produce(sam.explorers[4].U.V.viout) for sam in samplers] # different viout across NRSTSamplers, same within 

using Turing,StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end
c3 = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
extrema(c3[:,1,1])
mean(c3[:,1,1])
#########################################################################################
# Outline of how sampling using Turing.HMC works
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
#	  f) ... 
# 3)   After "initialstep" returns, "mcmcsampling" proceeds using the regular "step!" function, possibly adapting.
#      At each step an "HMCTransition" is created, which holds the parameters in constrained space
#   https://github.com/TuringLang/Turing.jl/blob/b5fd7611e596ba2806479f0680f8a5965e4bf055/src/inference/hmc.jl#L30   
#      This is achieved by calling "tonamedtuple(vi)", which essentially does vi[@varname(v)] for each variable v.
#   https://github.com/TuringLang/DynamicPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L1022
#      This indexing is the part that implicitly carries out the invlink'ing for a linked vi 
#   https://github.com/TuringLang/DynamicPPL.jl/blob/1744ba7bbc9da5cd8e0d04ab66ba588030e30879/src/varinfo.jl#L911
#
#      Eventually, sampling finishes, at which point "bundle_samples" is called. this is specialized in Turing,
#      to construct their Chains object from the "samples", which for HMC is a vector of HMCTransitions. Therefore,
#      no need to work with invlink at this stage since the HMCTransitions already have the constrained values.
#   https://github.com/TuringLang/Turing.jl/blob/9087412bb574bc83eacd9301f7fa5892a839c666/src/inference/Inferen