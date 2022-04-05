###############################################################################
# TODO: transform this into an actual test

# ###############################################################################
# # check E^{b}[V] is accurately estimated
# # compare F' to 
# # - multithr touring with NRST : ✓
# # - singlethr touring with NRST: ✓
# # - serial sampling with NRST  : ✓
# # - IID sampling from truth    : ✓
# # - serial exploration kernels : ✓
# # - parlel exploration kernels : ✓
# ###############################################################################

# cvd_pal = :tol_bright
# plot(F',0.,1., label="Theory", palette = cvd_pal) # ground truth

# # parallel NRST
# aggV = similar(ns.np.c)
# for (i, xs) in enumerate(resexact[:xarray])
#     aggV[i] = mean(ns.np.V.(xs))
# end
# plot!(ns.np.betas, aggV, label="MT-Tour", palette = cvd_pal)

# # iid sampling
# meanv(b) = mean(ns.np.V.(eachcol(rand(MultivariateNormal(mu(b), sbsq(b)*I(d)),1000))))
# plot!(ns.np.betas, meanv.(ns.np.betas), label="MC", palette = cvd_pal)

# # use the explorers to approximate E^{b}[V]: single thread
# for i in eachindex(aggV)
#     if i==1
#         aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
#     else        
#         traceV = similar(aggV, 5000)
#         NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
#         aggV[i] = mean(traceV)
#     end
# end
# plot!(ns.np.betas, aggV, label="SerMH", palette = cvd_pal)

# # use the explorers to approximate E^{b}[V]: multithread
# Threads.@threads for i in eachindex(aggV)
#     if i==1
#         aggV[1] = mean(NRST.tune!(ns.explorers[1], ns.np.V,nsteps=500))
#     else        
#         traceV = similar(aggV, 5000)
#         NRST.run_with_trace!(ns.explorers[i], ns.np.V, traceV)
#         aggV[i] = mean(traceV)
#     end
# end
# plot!(ns.np.betas, aggV, label="ParMH", palette = cvd_pal)

# # serial NRST: no tours involved
# xtrace, iptrace = NRST.run!(ns, nsteps=10000)
# aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
# nvisits = zeros(Int, length(aggV))
# for (n, ip) in enumerate(eachcol(iptrace))
#     nvisits[ip[1] + 1] += 1
#     aggV[ip[1] + 1]    += ns.np.V(xtrace[n])
# end
# aggV ./= nvisits
# plot!(ns.np.betas, aggV, label="SerNRST", palette = cvd_pal)

# # single thread NRST by tours
# xtracefull = Vector{typeof(ns.x)}(undef, 0)
# iptracefull = Matrix{Int}(undef, 2, 0)
# ntours = 1000
# for k in 1:ntours
#     xtrace, iptrace = NRST.tour!(ns)
#     append!(xtracefull, xtrace)
#     iptracefull = hcat(iptracefull, reduce(hcat, iptrace))
# end
# aggV = zeros(eltype(ns.np.c), length(ns.np.c)) # accumulate sums here, then divide by nvisits
# nvisits = zeros(Int, length(aggV))
# for (n, ip) in enumerate(eachcol(iptracefull))
#     nvisits[ip[1] + 1] += 1
#     aggV[ip[1] + 1]    += ns.np.V(xtracefull[n])
# end
# aggV ./= nvisits
# plot!(ns.np.betas, aggV, label="ST-Tour", palette = cvd_pal)

