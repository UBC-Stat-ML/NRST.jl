###############################################################################
# mrnatrans
###############################################################################

using NRST
using DelimitedFiles
using Distributions
using Random
#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct MRNATrans{TF<:AbstractFloat} <: NRST.TemperedModel
    as::Vector{TF}
    bs::Vector{TF}
    bma::Vector{TF} # b-a
    ts::Vector{TF}
    ys::Vector{TF}
end
MRNATrans(as,bs,ts,ys) = MRNATrans(as,bs,bs .- as,ts,ys)
function MRNATrans()
    MRNATrans(
        [-2., -5., -5., -5., -2.],
        [ 1.,  5.,  5.,  5.,  2.],
        mrna_trans_load_data()...
    )
end

function mrna_trans_load_data()
    dta = readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/transfection.csv", ',')
    ts  = Float64.(dta[2:end,1])
    ys  = Float64.(dta[2:end,3])
    return ts, ys
end

# methods for the prior
# if x~U[a,b] and y = f(x) = 10^x = e^{log(10)x}, then
# P(Y<=y) = P(10^X <= y) = P(X <= log10(y)) = F_X(log(y))
# p_Y(y) = d/dy P(Y<=y) = p_X(log(y)) 1/y = [ind{10^a<= y <=10^b}/(b-a)] [1/y]
# which is the reciprocal distribution or logUniform, so it checks out
function NRST.Vref(tm::MRNATrans{TF}, x) where {TF}
    vr = zero(TF)
    for (i,x) in enumerate(x)
        if x < tm.as[i] || x > tm.bs[i] 
            vr = TF(Inf)
            break
        end
    end
    return vr
end
function Random.rand!(tm::MRNATrans, rng, x)
    for (i, a) in enumerate(tm.as)
        x[i] = a + rand(rng) * tm.bma[i]
    end
    return x
end
function Base.rand(tm::MRNATrans{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, length(tm.as)))
end

# method for the likelihood potential
function NRST.V(tm::MRNATrans{TF}, x) where {TF}
    t₀  = 10^x[1]
    km₀ = 10^x[2]
    β   = 10^x[3]
    δ   = 10^x[4]
    σ   = 10^x[5]
    δmβ = δ - β
    acc = zero(TF)
    for (n, t) in enumerate(tm.ts)
        tmt₀ = t - t₀
        μ    = (km₀ / δmβ) * (-expm1(-δmβ * tmt₀)) * exp(-β*tmt₀)
        isfinite(μ) || (μ = TF(1e4))
        acc -= logpdf(Normal(μ, σ), tm.ys[n])
    end
    return acc
end

rng = SplittableRandom(8371)
tm  = MRNATrans()
ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            # tune=false
);
ns.np.xplpars .= [(sigma=rand(),) for _ in 1:ns.np.N]
nrpt = NRST.NRPTSampler(ns);
NRST.params.(NRST.get_xpls(nrpt)) == ns.np.xplpars
NRST.tune_explorers!(ns.np,nrpt,rng)
NRST.run!(nrpt, rng, 3200);
NRST.params.(NRST.get_xpls(nrpt)) == ns.np.xplpars
per   = NRST.get_perm(nrpt) # per[i]  = level of the ith machine (per[i] ∈ 0:N). note that machines are indexed 1:(N+1)
sper  = sortperm(per)  # sper[i] = id of the machine that is in level i-1 (sper[i] ∈ 1:(N+1))
[nrpt.nss[i].ip[1] for i in sper[2:end]]

any(iszero, [1e-324])

using Plots
using Plots.PlotMeasures: px
res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

###############################################################################
# end
###############################################################################
