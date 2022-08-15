using NRST
using NRSTExp.ExamplesGallery
using Plots
using Plots.PlotMeasures: px
using ColorSchemes: okabe_ito

tm = MvNormalTM(32,4.,2.);
r = SplittableRandom(1313)
ns, ts = NRSTSampler(
    tm,
    r,
    N = 11,
    verbose = true,
    do_stage_2 = false,
    maxcor = 0.8
);
copyto!(ns.np.c, free_energy(tm, ns.np.betas)); # use exact free energy
res   = parallel_run(ns, r, ntours=10000, keep_xs=false);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)