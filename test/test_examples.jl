using NRST
using NRST.ExamplesGallery
using Plots
using Plots.PlotMeasures: px

tm  = XYModel(8)
rng = SplittableRandom(0x0123456789abcdfe)
ns, ts = NRSTSampler(
    tm,
    rng,
    N = 12,
    verbose = true
)
res   = parallel_run(ns, rng, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

# Λ=5.27