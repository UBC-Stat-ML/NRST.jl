# ###############################################################################
# # numerical computation of variance of tourlength
# ###############################################################################

# # # build transition matrix for the index process
# # # convention:
# # # P = [(probs move up)           (probs of reject move up)]
# # #     [(probs of reject move dn) (probs move dn)          ]
# # # fill order: clockwise
# # # then, atom (0,-) is at index nlvls+1
# # using SparseArrays
# # using LinearAlgebra
# # R = NRST.rejrates(res)
# # nlvls   = size(R,1)
# # N       = nlvls-1
# # nstates = 2nlvls
# # IP = vcat(1:N, 1:nlvls, (nlvls+2):nstates, (nlvls+1):nstates)
# # JP = vcat(2:nstates, (nlvls+1):(nstates-1), 1:nlvls) # skip zero at (nlvls,nlvls+1). quadrants 1+2 combined, 
# # VP = vcat(1 .- R[1:(end-1),1], R[:,1], 1 .- R[2:end,2], R[:,2])
# # # any(iszero,VP)
# # P = sparse(IP,JP,VP,nstates,nstates)
# # # show(IOContext(stdout, :limit=>false), MIME"text/plain"(), P)
# # # all(isequal(1),sum(P,dims=2))

# # # get stationary distribuion <=> get left-nullspace of P-I 
# # # <=> get right-nullspace of P'-I
# # # <=> get QR decomposition of P-I # less accurate
# # # π∞ = qr(P-I).Q[:,end]
# # π∞ = nullspace(Matrix(P'-I))[:,1]
# # π∞ = π∞ / sum(π∞)

# # # using formulas from Kemeny & Snell (1960, Ch. 3) for absorbing MCs
# # # build Q matrix by dropping the row and column for the atom
# # qidxs = setdiff(1:nstates, nlvls+1)
# # Q = P[qidxs,qidxs]
# # # show(IOContext(stdout, :limit=>false), MIME"text/plain"(), Q)
# # # sum(x->x>0, 1 .- sum(Q,dims=2)) == 2 # must be only two ways to get to atom: 1) reject up move from (0,+), or 2) accept dn move from (1,-)

# # # fundamental matrix: F_{i,j} = expected number of visits to state j absorption when chain is started at i (i,j transient)
# # F = inv(Matrix(I - Q))

# # # expected number of visits to a level when started at (0,+), regardless of direction
# # # need to add last step for atom (0,+), which is not counted because its modelled as absorbing
# # F[1,1:nlvls] + pushfirst!(F[1,(nlvls+1):(nstates-1)],1.)
# # vec(sum(res.visits,dims=2) / NRST.get_ntours(res))# compare to sample estimates

# # # expected length of sojourn in the transient states, starting from any such state (Thm 3.3.5)
# # # note: expected tourlength = 𝔼τ[1]+1, since this does not count the last step at the atom
# # 𝔼τ = sum(F,dims=2)
# # (𝔼τ[1]+1, inv(π∞[nlvls+1]), 2(N+1)) # compare to 𝔼tourlength obtained from stationary distribuion and perfect tuning. 1st 2 almost exact if nullspace is used for π∞, but very different if QR method used

# # # variance of the tour length (Thm 3.3.5)
# # # note: var(τ)=var(τ+1) so the same formula applies
# # 𝔼τ² = (2F-I)*𝔼τ
# # 𝕍τ  = 𝔼τ² .- (𝔼τ .^2)
# # 𝕍τ[1] # variance of tourlength

# ###############################################################################
# # visual diagnostics
# ###############################################################################

# const DEF_PAL = seaborn_colorblind # default palette

# function diagnostics(ns::RegenerativeSampler, res::TouringRunResults)
#     N      = ns.np.N
#     ntours = get_ntours(res)

#     # occupancy rates
#     pocc = plot(
#         0:N, vec(sum(res.visits,dims=2) / ntours),
#         ylims = (0., Inf),
#         palette=DEF_PAL, label = "", xlabel = "Level", 
#         ylabel = "Average number of visits per tour" 
#     )

#     # rejection rates
#     R = rejrates(res)
#     averej  = 0.5*(R[1:(end-1),1]+R[2:end,2])
#     prrs = plot(
#         0:(N-1), R[1:(end-1),1],#push!(R[1:(end-1),1],NaN),
#         ylims = (0., Inf), legend = :bottomright, linestyle = :dash,
#         palette=DEF_PAL, label = "Up-from", xlabel = "Level", 
#         ylabel = "Rejection probability"#, legend_foreground_color = nothing
#     )
#     plot!(prrs,
#         0:(N-1), R[2:end,2],#pushfirst!(R[2:end,2],NaN),
#         palette=DEF_PAL, label = "Down-to", linestyle = :dash
#     )
#     plot!(prrs,
#         0:(N-1), averej, palette=DEF_PAL, label = "Average"
#     )

#     # plot explorers acceptance probabilities and nexpls
#     xplap = res.xplapac ./ vec(sum(res.visits, dims=2))[2:end]
#     pexpap = plot(
#         xplap, label="", xlabel="Level", ylabel="Explorers acceptance prob.",
#         palette = DEF_PAL
#     )
#     pnexpl = plot(
#         ns.np.nexpls, label="", xlabel="Level", palette = DEF_PAL,
#         ylabel="Exploration length per NRST step"
#     )

#     # Lambda Plot
#     betas = ns.np.betas
#     f_Λnorm, _, Λs = gen_lambda_fun(betas, averej, ns.np.log_grid)
#     plam = plot_lambda(β->((x = ns.np.log_grid ? floorlog(β) : β);Λs[end]*f_Λnorm(x)),betas,"")

#     # Plot of the log-partition function
#     lpart = log_partition(ns.np, res);
#     plp = plot(
#         betas, lpart,# ribbon = (lp_df[:,1]-lp_df[:,2], lp_df[:,3]-lp_df[:,1]),
#         palette=DEF_PAL, legend = :bottomright,
#         xlabel = "β", ylabel = "log(Z(β)/Z(0))", label = ""
#     )

#     # histogram of tour lengths
#     tlens  = tourlengths(res)
#     ltlns  = log10.(tlens)
#     lticks = make_log_ticks(ltlns)
#     ptlens = histogram(
#         ltlns, normalize=true, palette = DEF_PAL, #xaxis = :log10, xlims = extrema(tourlens),
#         xlabel = "Tour length", ylabel = "Density", label = "",
#         xticks = (collect(lticks), ["10^{$e}" for e in lticks])
#     );
#     meantlen = mean(tlens)
#     vline!(ptlens,
#         [log10(meantlen)], palette = DEF_PAL, linestyle = :dot,
#         label = "Mean=$(round(meantlen,digits=1))", linewidth = 4
#     )

#     # Density plots for V
#     colorgrad = cgrad([DEF_PAL[1], DEF_PAL[2]], range(0.,1.,N+1));
#     vrange = extrema(Base.Flatten([extrema(trV) for trV in res.trVs[((N+1)÷2):end]]))
#     vwidth = vrange[2] - vrange[1]
#     pdens = density(
#         res.trVs[1], color = colorgrad[1], label="", xlabel = "V",
#         ylabel = "Density", xlims =(vrange[1] - 0.05vwidth, vrange[2] + 0.05vwidth)
#     )
#     for i in 2:(N+1)
#         density!(pdens, res.trVs[i], color = colorgrad[i], label="")
#     end

#     # ESS/ntours versus toureff for the sine function
#     # note: TE bound is for true ESS and true TE. Sample estimates might not work. 
#     # periodic+continuous (thus bounded) functions of V are nice diagnostic tools
#     # because they don't care about the magnitude or scale of V, and much less 
#     # about the type of x. Therefore, they can be applied to any problem.
#     inf_df = inference_on_V(res, h = sin, at = 0:N)
#     pvess  = plot(
#         ns.np.betas, inf_df[:,"rESS"], xlabel = "β", 
#         label = "rESS", palette = DEF_PAL
#     )
#     plot!(pvess, ns.np.betas, res.toureff, label="Tour Eff.")
#     hline!(pvess, [1.], linestyle = :dash, label="")

#     # TODO: the following should be in its own plot
#     # plot of the trace of the (1st comp) of the index process
#     # plot_trace_iproc(res)
#     return (
#         occ=pocc, rrs=prrs, expap=pexpap, nexpl=pnexpl, lam=plam, lpart=plp,
#         tourlens=ptlens, dens=pdens, esspertour=pvess
#     )
# end

# #######################################
# # utils
# #######################################

# # utility for creating the Λ plot
# function plot_lambda(Λ,bs,lab)
#     c1 = DEF_PAL[1]
#     c2 = DEF_PAL[2]
#     p = plot(
#         Λ, 0., 1., label = "", legend = :bottomright,
#         xlim=(0.,1.), color = c1, grid = false, ylim=(0., Λ(bs[end])),
#         xlabel = "β", ylabel = "Λ(β)"
#     )
#     plot!(p, [0.,0.], [0.,0.], label=lab, color = c2)
#     for (i,b) in enumerate(bs[2:end])
#         y = Λ(b)
#         plot!(p, [b,b], [0.,y], label="", color = c2)                  # vertical segments
#         plot!(p, [0,b], [y,y], label="", color = c1, linestyle = :dot) # horizontal segments
#     end
#     p
# end

# # utility to make nice log ticks
# function make_log_ticks(lxs::AbstractVector{<:Real}, idealdiv::Int=5)
#     lmin, lmax   = extrema(lxs)
#     tlmin, tlmax = ceil(Int,lmin), floor(Int,lmax)
#     width        = tlmax-tlmin
#     if width == 0
#         return tlmin:tlmax
#     end
#     candidates   = 1:width 
#     divisors     = candidates[findall([width % c == 0 for c in candidates])]
#     bestdiv      = divisors[argmin(abs.(divisors .- idealdiv))] # ideal div implies div+1 actual ticks  
#     return tlmin:(width÷bestdiv):tlmax
# end

