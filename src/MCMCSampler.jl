abstract type MCMCSampler{T,TI<:Int,TF<:AbstractFloat} end

# run for fixed number of steps
function run!(mc::MCMCSampler, rng::AbstractRNG, tr::NRSTTrace; kwargs...)
    for n in eachindex(tr)
        save_pre_step!(mc,tr,n; kwargs...)
        save_post_step!(mc,tr,n,step!(mc, rng)...)
    end
end
function run!(mc::MCMCSampler, rng::AbstractRNG; nsteps::Int, kwargs...) 
    tr = get_trace(mc, nsteps)
    run!(mc, rng, tr; kwargs...)
    return tr
end