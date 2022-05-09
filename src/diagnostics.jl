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
        0:N, push!(rejrates[1:(end-1),1],NaN),
        ylims = (0., Inf), legend = :outertopright, linestyle = :dash,
        palette=DEF_PAL, label = "Up", xlabel = "Level", 
        ylabel = "Rejection probability", legend_foreground_color = nothing
    )
    plot!(prrs,
        0:N, pushfirst!(rejrates[2:end,2],NaN),
        palette=DEF_PAL, label = "Down", linestyle = :dash
    )
    plot!(prrs,
        0.5:(N-0.5), 0.5*(rejrates[1:(end-1),1]+rejrates[2:end,2]),
        palette=DEF_PAL, label = "Mean"
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
        linewidth = 4, label = "2N+2=$(2*(N+1))"
    )

    # Density plots for V
    colorgrad = cgrad([DEF_PAL[1], DEF_PAL[2]], range(0.,1.,N+1));
    minV   = minimum(minimum.(res.trVs))
    ltrVs  = [log10.(100eps() .+ trV .- minV) for trV in res.trVs]
    lticks = make_log_ticks(vcat(ltrVs...))
    pdens  = density(
        ltrVs[1], color=colorgrad[1], label="", xlabel = "V", ylabel = "Density",
        xticks = (collect(lticks), ["10^{$e}" for e in lticks])
    )
    for i in 2:(N+1)
        density!(pdens,ltrVs[i], color = colorgrad[i], label="")
    end


    # # line plot comparing cumsum(sort(tourlens)) v. sort(tourlens); i.e., the "cummax"
    # stl   = sort(tourlens)
    # csstl = cumsum(stl)
    # pcs   = plot(
    #     csstl, xlabel="Cumulative tours completed", ylabel="Number of NRST steps",
    #     label = "Sum", palette = DEF_PAL, yaxis=:log
    # )
    # plot!(pcs, stl, label = "Max")

    # # ESS/ntours for V versus toureff
    # pvess = plot(
    #     0:N, V_df[:,"ESS"] ./ ntours(res), xlabel="Level", label="ESS/#tours",
    #     palette = DEF_PAL
    # )
    # plot!(pvess, 0:N, res.toureff, label="TE")
    # hline!(pvess, [1.], linestyle = :dash, label="")

    return (pocc,prrs,plam,plp,ptlens,pdens)#,pcs,pvess)
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

function make_log_ticks(lxs)
    lmin, lmax   = extrema(lxs)
    tlmin, tlmax = ceil(Int,lmin), floor(Int,lmax)
    width        = tlmax-tlmin
    candidates   = 1:width 
    divisors     = candidates[findall([width % c == 0 for c in candidates])]
    bestdiv      = divisors[argmin(abs.(divisors.-5))]
    return tlmin:(width÷bestdiv):tlmax
end
