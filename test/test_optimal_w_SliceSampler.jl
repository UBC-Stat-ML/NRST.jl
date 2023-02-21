###############################################################################
# find best w for SliceSamplers
# conclusion: all w achieve same autocor and sqdist, but for
# 1) SliceSamplerStepping: w in (3σ, 5σ) is cheapest (~5 V evals per sample)
# 1) SliceSamplerDoubling: w in (12σ, 16σ) is cheapest (~6 V evals per sample)
###############################################################################

using Distributions, DynamicPPL, Plots, StatsBase
using SplittableRandoms
using NRST

const σ = Ref(1.0)
@model function ToyModel()
    y1 ~ Normal(0., σ[])
end
const tm  = NRST.TuringTemperedModel(ToyModel())
const rng = SplittableRandom(1)
const x   = rand(tm,rng)
const ps  = NRST.potentials(tm,x)
const ws  = 2 .^ range(-5,5,30)
const sqds= similar(ws)
const ss  = NRST.SliceSamplerStepping(
    tm, x, Ref(1.0), Ref(ps[1]), Ref(0.0), Ref(ps[1]), Ref(10.0), 2^20
);
function get_ac_nvs()
    nsim= 40000
    vs  = similar(ws, nsim)
    nvs = 0
    for i in 1:nsim
        nvs  += last(NRST.step!(ss,rng))
        vs[i] = ss.curVref[]
    end
    first(autocor(vs, 1:1)), nvs
end

acs = similar(ws)
nvs = Vector{Int}(undef, length(ws))
for (i,w) in enumerate(ws)
    ss.w[] = w
    print("Set w=$(ss.w[]). Sampling...")
    ac, nv = get_ac_nvs()
    println("done!")
    acs[i]=ac
    nvs[i]=nv
end
_,iopt = findmin(nvs)
plot(ws,nvs)
vline!([ws[iopt]])
2e5/4e4

copyto!(ws,2 .^ range(1,3,30))
σ[]=1.0
