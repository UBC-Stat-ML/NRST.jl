#########################################################################################
# interfacing with the Turing.jl environment
#########################################################################################

using Random, Distributions, DynamicPPL, NRST, Plots, StatsBase

# lognormal prior for variance, normal likelihood
@model function Lnmodel(x)
    s  ~ LogNormal()
    x .~ Normal(0.,s)
end
lnmodel = Lnmodel(randn(30)) # -.1234
betas = range(0,1,30) .^(2)
ns=NRSTSampler(
    lnmodel, # use the Turing model to build potentials and prior sampling
    betas,
    50,      # nexpl
    true     # tune c using mean energy
);
@code_warntype run!(ns,nsteps=50000);
res = post_process(run!(ns,nsteps=50000));
plot(0:ns.np.N, vec(sum(res.visits,dims=2)))
tune_c!(ns,res);

# parallel
samplers = copy_sampler(ns, nthreads = 4);
par_res = run!(samplers, ntours = 5000);
plot(0:ns.np.N, vec(sum(par_res.visits,dims=2)))
ns.np.fns.viout
[get_num_produce(s.np.fns.viout) for s in samplers] # they should be different
par_res.toureff

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