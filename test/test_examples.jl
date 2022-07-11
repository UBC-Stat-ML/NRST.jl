
# ## Notes on the results
# ### Inspecting within and between-group std. devs.
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
pcover = scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    markeralpha = min(1., max(0.08, 1000/size(X,2))),
    ylabel="σ: within-group std. dev.", label=""
)
