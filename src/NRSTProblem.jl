# using UnPack,BenchmarkTools,Distributions,StatsPlots,Printf,Loess

# include("ExplorationKernels.jl"); # load code for explorer kernels

# union of concrete types
# useful for having vectors of mixed types (as long as <=4 types)
# see https://stackoverflow.com/a/58539098/5443023
# also https://docs.julialang.org/en/v1/manual/types/#citeref-1
const ContinuousSampler = Union{IIDSampler, MHSampler}

###############################################################################
# struct defining an NRST inference problem 
###############################################################################

struct NRSTProblem{F,G,H,K<:AbstractFloat,A<:AbstractVector{K},B<:AbstractVector{<:ExplorationKernel}}
    V::F # energy Function
    Vref::G # energy of reference distribution
    randref::H # produces independent sample from reference distribution
    betas::A # vector of tempering parameters (length N)
    c::A # vector of parameters for the pseudoprior
    explorers::B # vector length N of exploration kernels (note use of concrete eltype)
end

# initialize vector of exploration kernels: continous case
function init_explorers(V,Vref,randref,betas,xinit::AbstractVector{<:AbstractFloat})
    A = Vector{ContinuousSampler}(undef, length(betas))
    A[1] = IIDSampler(Vref,randref)
    for i in 2:length(betas)
        beta = betas[i] # better to extract the beta, o.w. the closure grabs the whole vector
        A[i] = MHSampler(x->(Vref(x) + beta*V(x)),xinit)
    end
    return A
end

# simple outer constructor
function NRSTProblem(V,Vref,randref,betas,xinit)
    c = similar(betas)
    explorers = init_explorers(V,Vref,randref,betas,xinit) 
    NRSTProblem(V,Vref,randref,betas,c,explorers)
end

# test
# likelihood: N((1,1), I), reference: N(0, 4I)
# => -log-target = K + 1/2[(x-1)^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^T(x-1)-1^T(x-1) + 1/4x^Tx]
# = K + 1/2[x^Tx - 21^Tx + 1/4x^Tx]
# = K + 1/2[(5/4)x^Tx - 2(5/4)(4/5)1^Tx]
# = K + 1/2(5/4)[x^Tx - 2(4/5)1^Tx]
# = K + 1/2(5/4)(x-4/5)^T(x-4/5)
# => target: N((4/5, 4/5), (4/5)I)
# np = NRSTProblem(
#     x->(0.5sum(abs2,x .- [1.,1.])), # likelihood: N((1,1), I)
#     x->(0.125sum(abs2,x)), # reference: N(0, 4I)
#     () -> 2randn(2),
#     collect(range(0,1,length=9)), # uniform grid with N=8
#     [1.,1.]
# );
# pal = palette([:purple, :green], length(np.betas))
# xrange = -2:0.1:2; lenx = length(xrange)
# Z = ((x1,x2) -> np.explorers[1].U([x1,x2])).(xrange,xrange')
# ind = round.(Int,lenx*lenx*[0.005,0.1])
# p=plot(legend = :none,aspect_ratio=1)
# plot!(p,[0,4/5],seriestype=:vline,linestyle=:dot,linewidth=2,linecolor=pal[1])
# plot!(p,[0,4/5],seriestype=:hline,linestyle=:dot,linewidth=2,linecolor=pal[1])
# plot!(p,[0,4/5],seriestype=:vline,linestyle=:dot,linewidth=2,linecolor=pal[end])
# plot!(p,[0,4/5],seriestype=:hline,linestyle=:dot,linewidth=2,linecolor=pal[end])
# for (i,expl) in enumerate(np.explorers)
#     Z = ((x1,x2) -> expl.U([x1,x2])).(xrange,xrange')
#     contour!(p,xrange,xrange, Z,levels=Z[sortperm(vec(Z))[ind]],
#     linecolor=pal[i])
# end
# p

# tune all explorers' parameters in parallel
# also, use their output to fit c using mean(V) method. for i>=0,
# c[i] = int_0^beta_i db E^{b}[V]
# then c[0]=0, and for i>=1 use trapezoidal approx
# ≈ 0.5 sum_{j=1}^i (E^{beta_j}[V]+E^{beta_j-1}[V])(beta_j-beta_{j-1})
# ≈ 0.5sum_{j=1}^i (mean^{beta_j}[V]+mean^{beta_j-1}[V])(beta_j-beta_{j-1})
# recursively,
# c[i] = c[i-1] + 0.5(mean^{beta_i}[V]+mean^{beta_i-1}[V])(beta_i-beta_{i-1})
function initial_tuning!(np::NRSTProblem,nsteps::Int)
    # nsteps=5000
    @unpack c,explorers,betas,V = np
    meanV = similar(betas)
    Threads.@threads for i in eachindex(meanV)
        meanV[i] = tune!(explorers[i],V,nsteps=nsteps)
    end
    # use LOESS smoothing to remove kinks
    copyto!(meanV,predict(loess(betas, meanV),betas)) # code_warntype: predict is not type stable!
    # compute trapezoidal approx. cache meanV, betas to avoid double access
    c[1] = 0; oldmv=meanV[1]; oldb=betas[1]
    for i in 2:length(meanV)
        newmv=meanV[i]; newb=betas[i]
        c[i] = c[i-1] + 0.5(oldmv+newmv)*(newb-oldb)
        oldmv=newmv; oldb=newb
    end
end

# # test
# @code_warntype initial_tuning!(np,5000) # predict not type stable!
# initial_tuning!(np,5000)
# plot(np.betas, np.c)
