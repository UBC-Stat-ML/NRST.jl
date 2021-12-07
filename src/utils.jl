# trapezoidal approximation
# want: c[i] = c(x[i]) = int_0^x_i dx f(x)
# then c[0]=0, and for i>=1 use
# c[i] ≈ sum_{j=1}^i 0.5(f(x_j) + f(x_{j-1}))(x_j - x_{j-1})
# Or recursively,
# c[i] = c[i-1] + 0.5(f(x_j) + f(x_{j-1}))(x_j - x_{j-1})
function trpz_apprx!(c::T,xs::T,ys::T) where {K<:AbstractFloat,T<:AbstractVector{K}}
    @assert length(c) == length(xs) == length(ys)
    c[1] = zero(K)
    oldx = xs[1]
    oldy = ys[1]
    for i in 2:length(xs)
        x = xs[i]
        y = ys[i]
        c[i] = c[i-1] + 0.5(y + oldy) * (x - oldx)
        oldx = x
        oldy = y
    end
end