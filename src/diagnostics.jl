###############################################################################
# visual diagnostics
###############################################################################

const DEF_PAL = seaborn_colorblind # default palette

function diagnostics(ns::NRSTSampler, res::ParallelRunResults)
    N = ns.np.N

    # occupancy rates
    pocc = plot(
        0:N, vec(sum(res.visits,dims=2) / ntours(res)),
        ylims = (0., Inf),
        palette=DEF_PAL, label = "", xlabel = "Level", 
        ylabel = "Average number of visits per tour" 
    )

    # rejection rates
    rejrates=res.rpacc ./ res.visits
    prrs = plot(
        0:(N-1), rejrates[1:(end-1),1],#push!(rejrates[1:(end-1),1],NaN),
        ylims = (0., Inf), legend = :bottomright, linestyle = :dash,
        palette=DEF_PAL, label = "Up-from", xlabel = "Level", 
        ylabel = "Rejection probability"#, legend_foreground_color = nothing
    )
    plot!(prrs,
        0:(N-1), rejrates[2:end,2],#pushfirst!(rejrates[2:end,2],NaN),
        palette=DEF_PAL, label = "Down-to", linestyle = :dash
    )
    plot!(prrs,
        0:(N-1), 0.5*(rejrates[1:(end-1),1]+rejrates[2:end,2]),
        palette=DEF_PAL, label = "Average"
    )

    # plot explorers acceptance probabilities and nexpls
    xplap = res.xplapac ./ vec(sum(res.visits, dims=2))[2:end]
    pexpap = plot(
        xplap, label="", xlabel="Level", ylabel="Explorers acceptance prob.",
        palette = DEF_PAL
    )
    pnexpl = plot(
        ns.np.nexpls, label="", xlabel="Level", palette = DEF_PAL,
        ylabel="Exploration length per NRST step"
    )

    # Lambda Plot
    betas = ns.np.betas
    f_Λnorm, Λsnorm, Λs = NRST.get_lambda(betas, rejrates)
    plam = plot_lambda(β->Λs[end]*f_Λnorm(β),betas,"")

    # Plot of the log-partition function
    lpart = log_partition(ns, res);
    plp = plot(
        betas, lpart,# ribbon = (lp_df[:,1]-lp_df[:,2], lp_df[:,3]-lp_df[:,1]),
        palette=DEF_PAL, legend = :bottomright,
        xlabel = "β", ylabel = "log(Z(β)/Z(0))", label = ""
    )

    # histogram of tour lengths
    ltlns  = log10.(tourlengths(res))
    lticks = make_log_ticks(ltlns)
    ptlens = histogram(
        ltlns, normalize=true, palette = DEF_PAL, #xaxis = :log10, xlims = extrema(tourlens),
        xlabel = "Tour length", ylabel = "Density", label = "",
        xticks = (collect(lticks), ["10^{$e}" for e in lticks])
    );
    vline!(ptlens,
        [log10(2*(N+1))], palette = DEF_PAL, linestyle = :dot,
        label = "2N+2=$(2*(N+1))", linewidth = 4
    )

    # Density plots for V
    colorgrad = cgrad([DEF_PAL[1], DEF_PAL[2]], range(0.,1.,N+1));
    vrange = extrema(Base.Flatten([extrema(trV) for trV in res.trVs[((N+1)÷2):end]]))
    vwidth = vrange[2] - vrange[1]
    pdens = density(
        res.trVs[1], color = colorgrad[1], label="", xlabel = "V",
        ylabel = "Density", xlims =(vrange[1] - 0.05vwidth, vrange[2] + 0.05vwidth)
    )
    for i in 2:(N+1)
        density!(pdens, res.trVs[i], color = colorgrad[i], label="")
    end

    # ESS versus computational cost
    # compares serial v. parallel NRST and against idealizations of the index process
    pcs = plot_ess_time(res, Λs[end])

    # ESS/ntours versus toureff for a bounded function
    # note: TE bound is for true ESS and true TE. Sample estimates might not work. 
    # use logistic(Z(V)) where Z is standardization using median
    # looks weird but has benefits over other potential bounded functions
    # - V(x) is always defined for any model that got up to here
    # - Z(V) with median is defined even for non-integrable-under-reference V
    # - logistic(Z(V)) is bounded and has non trivial values in general -- ie,
    #   not always 0 or not always 1 -- since Z(V) cannot be too far of 0
    # The last condition fails for example for indicators like {V>v} for some
    # v, since the distribution of V changes radically between temperatures.
    mV = median(Base.Flatten(res.trVs))
    sV = median(abs.(Base.Flatten(res.trVs) .- mV))
    inf_df = inference_on_V(res, h = v -> logistic((v-mV)/sV), at = 0:N)
    pvess  = plot(
        ns.np.betas, inf_df[:,"ESS"] ./ ntours(res), xlabel = "β", 
        label = "ESS/#tours", palette = DEF_PAL
    )
    plot!(pvess, ns.np.betas, res.toureff, label="Tour Eff.")
    hline!(pvess, [1.], linestyle = :dash, label="")

    # TODO: the following should be in its own plot
    # plot of the trace of the (1st comp) of the index process
    # plot_trace_iproc(res)
    return (
        occ=pocc, rrs=prrs, expap=pexpap, nexpl=pnexpl, lam=plam, lpart=plp,
        tourlens=ptlens, dens=pdens, esscost=pcs, esspertour=pvess
    )
end

#######################################
# utils
#######################################

# utility for creating the Λ plot
function plot_lambda(Λ,bs,lab)
    c1 = DEF_PAL[1]
    c2 = DEF_PAL[2]
    p = plot(
        x->Λ(x), 0., 1., label = "", legend = :bottomright,
        xlim=(0.,1.), color = c1, grid = false, ylim=(0., Λ(bs[end])),
        xlabel = "β", ylabel = "Λ(β)"
    )
    plot!(p, [0.,0.], [0.,0.], label=lab, color = c2)
    for (i,b) in enumerate(bs[2:end])
        y = Λ(b)
        plot!(p, [b,b], [0.,y], label="", color = c2)                  # vertical segments
        plot!(p, [0,b], [y,y], label="", color = c1, linestyle = :dot) # horizontal segments
    end
    p
end

# utility to make nice log ticks
function make_log_ticks(lxs::AbstractVector{<:Real}, idealdiv::Int=5)
    lmin, lmax   = extrema(lxs)
    tlmin, tlmax = ceil(Int,lmin), floor(Int,lmax)
    width        = tlmax-tlmin
    if width == 0
        return tlmin:tlmax
    end
    candidates   = 1:width 
    divisors     = candidates[findall([width % c == 0 for c in candidates])]
    bestdiv      = divisors[argmin(abs.(divisors .- idealdiv))] # ideal div implies div+1 actual ticks  
    return tlmin:(width÷bestdiv):tlmax
end

# utility for the ess/time plot
# NOTE: as maxcor → 0, the tours lengths become more 
# consistent, so the NRST-par curve in the ESS/cost plot converges to the
# one for MC-par!
# TODO: add BouncyMC with imperfect tuning, using asymmetric, non-equi rejections
# that can be obtained from "res" (see rejrates in rejections plot)
function plot_ess_time(res::ParallelRunResults, Λ::AbstractFloat)
    resN   = N(res)
    resnts = ntours(res)
    labels = String[]
    xs     = Vector{typeof(Λ)}[]
    ys     = Vector{typeof(Λ)}[]

    # NRST
    tourls = tourlengths(res) 
    cuESS  = res.toureff[end]*(1:resnts)
    push!(xs, log10.(cumsum(tourls)))          # serial
    push!(ys, log10.(cuESS))
    push!(labels, "NRST-ser")
    push!(xs, log10.(accumulate(max, tourls))) # parallel
    push!(ys, log10.(cuESS))
    push!(labels, "NRST-par")
    
    # BouncyPDMP
    tourls, vNs  = run_tours!(BouncyPDMP(Λ), resnts)
    TE           = (sum(vNs) ^ 2) / (resnts*sum(abs2, vNs))
    cuESS        = TE*(1:resnts)
    scale_tourls = resN*tourls .+ 2.           # the shortest tour has n(0)=2 and the perfect roundtrip has n(2)=2N+2
    push!(xs, log10.(cumsum(scale_tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdCTSer")
    push!(xs, log10.(accumulate(max, scale_tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdCTPar")

    # BouncyMC: perfect tuning
    tourls, vNs = run_tours!(BouncyMC(Λ/resN,resN), resnts)
    TE = (sum(vNs) ^ 2) / (resnts*sum(abs2, vNs))
    cuESS = TE*(1:resnts)
    push!(xs, log10.(cumsum(tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdDTPerfSer")
    push!(xs, log10.(accumulate(max, tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdDTPerfPar")

    # BouncyMC: same tuning as NRST
    R = res.rpacc ./ res.visits
    tourls, vNs = run_tours!(BouncyMC(R), resnts)
    TE = (sum(vNs) ^ 2) / (resnts*sum(abs2, vNs))
    cuESS = TE*(1:resnts)
    push!(xs, log10.(cumsum(tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdDTActSer")
    push!(xs, log10.(accumulate(max, tourls)))
    push!(ys, log10.(cuESS))
    push!(labels, "IdDTActPar")

    # plot
    xlticks = make_log_ticks(collect(Base.Flatten([extrema(x) for x in xs])))
    ylticks = make_log_ticks(collect(Base.Flatten([extrema(y) for y in ys])))
    pcs     = plot(
        xlabel = "Computational time",
        ylabel = "ESS bound @ cold level",# palette = DEF_PAL, 
        legend = :bottomright,
        xticks = (xlticks, ["10^{$e}" for e in xlticks]),
        yticks = (ylticks, ["10^{$e}" for e in ylticks])
    )
    for (i,l) in enumerate(labels)
        c = (i-1)÷2
        s = i-2c
        plot!(
            pcs, xs[i], ys[i], label = l,# linewidth=2,
            linecolor = okabe_ito[c+1],
            linestyle = s==1 ? :dash : :solid
        )
    end
    return pcs
end

# function to create a plot of the trace of the (1st comp) of the index process
function plot_trace_iproc(res::ParallelRunResults)
    K   = min(floor(Int, 800/(2*ns.np.N+2)), ts.ntours) # choose K so that we see around a given num of steps
    len = sum(nsteps.(res.trvec[1:K]))
    is  = Vector{eltype(ns.ip)}(undef, len)
    l   = 1
    for tr in res.trvec[1:K]
        for ip in tr.trIP
            is[l] = ip[1]
            l += 1
        end
    end
    itop = findall(isequal(ns.np.N), is)[1:2:end]
    ibot = findall(isequal(zero(ns.np.N)), is)[2:2:end]
    piproc = plot(
        is, grid = false, palette = DEF_PAL, ylims = (0,1.04ns.np.N),
        xlabel = "Step", ylabel = "Index", label = "",
        left_margin = 15px, bottom_margin = 15px, size = (675, 225)
    )
    scatter!(piproc, itop .+ 0.5, [1.025ns.np.N], markershape = :dtriangle, label="")
    vline!(piproc, ibot .+ 0.5, linestyle = :dot, label="", linewidth=2)
    return piproc
end