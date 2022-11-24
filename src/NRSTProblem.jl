###############################################################################
# NRSTProblem: encapsulates all the specifics of the tempered problem
# TODO: rename to NRSTConfig eventually---its clearer about its purpose.
###############################################################################

struct NRSTProblem{TTM<:TemperedModel,K<:AbstractFloat,A<:Vector{K},TInt<:Int,TNT<:NamedTuple}
    tm::TTM              # a TemperedModel
    N::TInt              # number of states not counting reference (N+1 in total)
    betas::A             # vector of tempering parameters (length N+1)
    c::A                 # vector of parameters for the pseudoprior
    use_mean::Bool       # should we use "mean" (true) or "median" (false) for tuning c?
    reject_big_vs::Bool  # should we use rejection sampling from the reference to avoid the region {V=inf}
    log_grid::Bool       # should we tune the grid with beta in log scale. needed when π₀{V} = inf, since here the derivative of Λ at 0 is Inf.
    nexpls::Vector{TInt} # vector of length N with number of exploration steps adequate for each level 1:N
    xplpars::Vector{TNT} # vector of length N of named tuples, holding adequate parameters to use at each level 1:N
end

# copy constructor, allows replacing tm, but keeps everything else
function NRSTProblem(oldnp::NRSTProblem, newtm)
    NRSTProblem(
        newtm,oldnp.N,oldnp.betas,oldnp.c,oldnp.use_mean,oldnp.reject_big_vs,
        oldnp.log_grid,oldnp.nexpls,oldnp.xplpars
    )
end
