###############################################################################
# visual diagnostics
###############################################################################

const DEF_PAL = seaborn_colorblind # default palette

function diagnostics(
    ns::NRSTSampler,
    res::ParallelRunResults
    )

    # occupancy rates
    pocc = plot(
        0:ns.np.N, vec(sum(res.visits,dims=2) / ntours(res)),
        ylims = (0., Inf),
        palette=DEF_PAL, label = "", xlabel = "Level", 
        ylabel = "Average number of visits per tour" 
    )

    # rejection rates
    rejrates=res.rejecs ./ res.visits
    prrs = plot(
        0:ns.np.N, push!(rejrates[1:(end-1),1],NaN),
        ylims = (0., Inf), legend = :bottomright, linestyle = :dash,
        palette=DEF_PAL, label = "Up", xlabel = "Level", 
        ylabel = "Proportion of rejected proposals"
    )
    plot!(prrs,
        0:ns.np.N, pushfirst!(rejrates[2:end,2],NaN),
        palette=DEF_PAL, label = "Down", linestyle = :dash
    )
    plot!(prrs,
        0.5:(ns.np.N-0.5), 0.5*(rejrates[1:(end-1),1]+rejrates[2:end,2]),
        palette=DEF_PAL, label = "Average"
    )

    # Lambda Plot
    betas = ns.np.betas;
    Λnorm, _ = get_lambda(betas, res.rejecs ./ res.visits);
    plam = plot_lambda(Λnorm,betas,"")

    # Plot of the log-partition function
    lp_df = log_partition(ns, res);
    plp = plot(
        betas, lp_df[:,1], ribbon = (lp_df[:,1]-lp_df[:,2], lp_df[:,3]-lp_df[:,1]),
        palette=DEF_PAL, legend = :bottomright,
        xlabel = "β", ylabel = "log(Z(β))", label = ""
    )

    # Distribution of the tour lengths
    tourlens = tourlengths(res);
    ptlens = histogram(
        tourlens, normalize=true, palette = DEF_PAL,
        xlabel = "Tour length", ylabel = "Density", label = ""
    );
    N = ns.np.N
    vline!(ptlens,
        [2*(N+1)], palette = DEF_PAL, 
        linewidth = 4, label = "2N+2=$(2*(N+1))"
    )

    # Density plots for x[1]
    colorgrad = cgrad([DEF_PAL[1], DEF_PAL[2]], range(0.,1.,N+1));
    pdens = density(
        [x[1] for x in res.xarray[1]],color=colorgrad[1],label="",
        xlabel = "x[1]", ylabel = "Density"
    )
    for i in 2:(N+1)
        density!(pdens,
            [x[1] for x in res.xarray[i]], color = colorgrad[i], label=""
        )
    end

    return (pocc,prrs,plam,plp,ptlens,pdens)
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
        xlim=(0.,1.), ylim=(0.,1.), color = c1, grid = false,
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

