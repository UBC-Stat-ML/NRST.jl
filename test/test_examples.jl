using NRST
using DynamicPPL, Distributions
using Plots
using Plots.PlotMeasures: px

# lognormal prior, normal likelihood
@model function Lnmodel(x)
    s ~ LogNormal()
    x .~ Normal(0.,s)
end
model = Lnmodel(randn(30))
tm    = NRST.TuringTemperedModel(model);
rng   = SplittableRandom(4)
ns    = NRSTSampler(tm, rng, N=3);
res   = parallel_run(ns, rng, ntours = 10000);


# build transition matrix for the index process
# convention:
# P = [(probs move up)           (probs of reject move up)]
#     [(probs of reject move dn) (probs move dn)          ]
# fill order: clockwise
# then, atom (0,-) is at index nlvls+1
using SparseArrays
using LinearAlgebra
R = NRST.rejrates(res)
nlvls   = size(R,1)
N       = nlvls-1
nstates = 2nlvls
IP = vcat(1:N, 1:nlvls, (nlvls+2):nstates, (nlvls+1):nstates)
JP = vcat(2:nstates, (nlvls+1):(nstates-1), 1:nlvls) # skip zero at (nlvls,nlvls+1). quadrants 1+2 combined, 
VP = vcat(1 .- R[1:(end-1),1], R[:,1], 1 .- R[2:end,2], R[:,2])
# any(iszero,VP)
P = sparse(IP,JP,VP,nstates,nstates)
# show(IOContext(stdout, :limit=>false), MIME"text/plain"(), P)
# all(isequal(1),sum(P,dims=2))

# get stationary distribuion
π∞ = nullspace(Matrix(P'-I))[:,1]
π∞ = π∞ / sum(π∞)

# using formulas from Kemeny & Snell (1960, Ch. 3) for absorbing MCs
# build Q matrix by dropping the row and column for the atom
qidxs = setdiff(1:nstates, nlvls+1)
Q = P[qidxs,qidxs]
# show(IOContext(stdout, :limit=>false), MIME"text/plain"(), Q)
# sum(x->x>0, 1 .- sum(Q,dims=2)) == 2 # must be only two ways to get to atom: 1) reject up move from (0,+), or 2) accept dn move from (1,-)

# fundamental matrix: F_{i,j} = expected number of visits to state j absorption when chain is started at i (i,j transient)
F = inv(Matrix(I - Q))

# expected number of visits to a level when started at (0,+), regardless of direction
# need to add last step for atom (0,+), which is not counted because its modelled as absorbing
F[1,1:nlvls] + pushfirst!(F[1,(nlvls+1):(nstates-1)],1.)
vec(sum(res.visits,dims=2) / NRST.get_ntours(res))# compare to sample estimates

# expected length of sojourn in the transient states, starting from any such state (Thm 3.3.5)
# note: expected tourlength = 𝔼τ[1]+1, since this does not count the last step at the atom
𝔼τ = sum(F,dims=2)
(𝔼τ[1]+1, inv(π∞[nlvls+1])) # compare to 𝔼tourlength obtained from stationary distribuion 
(𝔼τ[1]+1, 2(N+1)) # compare 𝔼tourlength to perfect tuning theoretical value

# variance of the tour length (Thm 3.3.5)
# note: var(τ)=var(τ+1) so the same formula applies
𝔼τ² = (2F-I)*𝔼τ
𝕍τ  = 𝔼τ² .- (𝔼τ .^2)
𝕍τ[1] # variance of tourlength

###############################################################################
###############################################################################
using NRST
using DynamicPPL, Distributions
using LinearAlgebra
using Plots
using Plots.PlotMeasures: px
using DelimitedFiles
using Printf
using Random
@model function _HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                  # can't use improper prior in NRST
    θ  ~ MvNormal(fill(μ,J), τ²*I)
    for j in 1:J
        Y[:,j] ~ MvNormal(fill(θ[j], N), σ²*I)
    end
end
# Loading the data and instantiating the model
function HierarchicalModel()
    Y     = readdlm("/home/mbiron/Documents/RESEARCH/UBC_PhD/NRST/NRSTExp/data/simulated8schools.csv", ',', Float64)
    model = _HierarchicalModel(Y)
    return NRST.TuringTemperedModel(model)
end
tm    = HierarchicalModel();
rng   = SplittableRandom(4)
ns    = NRSTSampler(tm, rng, N=10);
res   = parallel_run(ns, rng, ntours = 32_768);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
pcover = scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    markeralpha = min(1., max(0.08, 1000/size(X,2))),
    ylabel="σ: within-group std. dev.", label=""
)