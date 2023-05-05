# convert a matrix to a vector of (row)vectors
rows2vov(M::AbstractMatrix) = [copy(r) for r in eachrow(M)]

# find odd number closest to a given real number
# used for determining window of running_median
# for this reason, the result is truncated at 3
#     2k-1 < x < 2k+1
#     2k-1 < x < 2(k+1)-1
#     k < (x+1)/2 < k+1
closest_odd(x) = max(3, 2round(Int, (x+1)/2)-1)

# find root for monotonic univariate functions
function monoroot(f, l::F, u::F; tol = 2000eps(F), maxit = typemax(Int)) where {F<:AbstractFloat}
    @assert l <= u
    fl = f(l)
    fu = f(u)
    if sign(fl) == sign(fu)                          # f monotone & same sign at both ends => no root in interval
        return abs(fl) < abs(fu) ? (l, fl) : (u, fu) # return the endpoint with f closest to zero
    end
    h  = l
    fh = fl                                          # init fh (so that it is available for the return outside the for loop)
    for i in 1:maxit
        h  = (l+u)/2
        fh = f(h)
        # println("i=$i: (l,h,u)=($l,$h,$u), (fl,fh,fu)=($fl,$fh,$fu)")
        if abs(fh) < tol
            return h, fh
        elseif sign(fl) == sign(fh)
            l  = h
            fl = fh
        else
            u  = h
            fu = fh
        end
    end
    return h, fh
end

# trapezoidal approximation
# want: c[i] = c(x[i]) = int_0^x_i dx f(x)
# then c[0]=0, and for i>=1 use
# c[i] ≈ sum_{j=1}^i 0.5(f(x_j) + f(x_{j-1}))(x_j - x_{j-1})
# Or recursively,
# c[i] = c[i-1] + 0.5(f(x_i) + f(x_{i-1}))(x_i - x_{i-1})
function trapez!(c::AbstractVector, xs::AbstractVector, ys::AbstractVector)
    @assert length(c) == length(xs) == length(ys)
    c[1] = zero(eltype(c))
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
function trapez(xs,ys)
    c = similar(xs)
    trapez!(c, xs, ys)
    return c
end

#######################################
# stepping stone
#######################################

# forward stepping stone: does not use samples at i=N+1
# Z_N/Z_0 = prod_{i=1}^N Z_i/Z_{i-1}
# <=> log(Z_N/Z_0) = sum_{i=1}^N log(Z_i/Z_{i-1})
# Now,
# Z_i = E^{0}[e^{-beta_i V}] 
# = int pi_0(dx) e^{-beta_i V(x)}
# = int [pi_0(dx) e^{-beta_{i-1} V(x)}] e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} int pi^{i-1}(dx) e^{-(beta_i-beta_{i-1}) V(x)}
# = Z_{i-1} E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# Hence
# Z_i/Z_{i-1} = E^{i-1}[e^{-(beta_i-beta_{i-1}) V(x)}]
# ≈ (1/S) sum_{n=1}^{S_{i-1}} e^{-(beta_i-beta_{i-1}) V(x_n)}, x_{1:S_{i-1}} ~ pi^{i-1}
# <=> log(Z_i/Z_{i-1}) ≈ -log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))
#  => log(Z_N/Z_0) = sum_{i=1}^N [-log(S_{i-1}) + logsumexp(-(beta_i-beta_{i-1}) V(x_{1:S_{i-1}}))]


# reverse stepping stone: does not use samples at i=0
# log(Z_N/Z_0) = -[log(Z_0/Z_N)] = - sum_{i=0}^{N-1} log(Z_i/Z_{i+1})
# but now use,
# Z_i = E^{0}[e^{-beta_i V}] 
# = int [pi_0(dx) e^{-beta_{i+1} V(x)}] e^{(beta_{i+1}-beta_i) V(x)}
# = Z_{i+1} int pi^{i+1}(dx) e^{(beta_{i+1}-beta_i) V(x)}
# = Z_{i+1} E^{i+1}[e^{(beta_{i+1}-beta_i) V(x)}]
# Hence
# Z_i/Z_{i+1} = E^{i+1}[e^{(beta_{i+1}-beta_i) V(x)}]
# ≈ (1/S_{i+1}) sum_{n=1}^{S_{i+1}} e^{(beta_{i+1}-beta_i) V_n},    V_{1:S_{i+1}} ~ pi^{i+1}
# <=> log(Z_i/Z_{i+1}) ≈ -log(S_{i+1}) + logsumexp((beta_{i+1}-beta_i) V_{1:S_{i+1}})
#  => log(Z_N/Z_0) = sum_{i=0}^{N-1} log(Z_{i+1}/Z_i) = - sum_{i=0}^{N-1} log(Z_i/Z_{i+1})
#    ≈ sum_{i=0}^{N-1} [log(S_{i+1}) - logsumexp((beta_{i+1}-beta_i) V_{1:S_{i+1}})]
#    = sum_{i=1}^{N} [log(S_i) - logsumexp((beta_i-beta_{i-1}) V_{1:S_i})]

# this function computes both forward and backward estimators simultaneously, and
# takes their weighted mean.
# note: stepping_stone is immune to V=Inf at i=1 (reference), see comments below.
const STEPSTONE_FWD_WEIGHT = Ref(0.5)
function stepping_stone!(
    zs::AbstractVector,
    bs::AbstractVector,
    trVs::Vector{TV}
    ) where {TF<:AbstractFloat, TV<:Vector{TF}}
    w     = STEPSTONE_FWD_WEIGHT[]
    onemw = one(TF) - w
    zs[1] = zero(TF)
    accf  = accb = zero(TF)
    llen  = log(length(trVs[1]))
    for i in 2:length(zs)
        db    = bs[i] - bs[i-1]
        accf += logsumexp(-db*trVs[i-1]) - llen # immune to V=inf at i=1 (reference) due to minus sign
        llen  = log(length(trVs[i]))
        accb += llen - logsumexp(db*trVs[i])    # immune to V=inf at i=1 because it does not use those samples
        zs[i] = w*accf + onemw*accb
    end
end
function stepping_stone(bs, trVs)
    zs = similar(bs)
    stepping_stone!(zs, bs, trVs)
    return zs
end

###############################################################################
# LOO for SmoothingSplines
###############################################################################

# # remove single element from vector using only 1 allocation
# # https://discourse.julialang.org/t/remove-an-element-from-an-array-without-changing-the-original/63103/3
# @views deleteelem(a::AbstractVector, i) = vcat(a[begin:i-1], a[i+1:end])

# function LOO(xs::AbstractVector, ys::AbstractVector; λs = 2. .^ ((-8):2:16))
#     L    = length(λs)
#     errs = zeros(L)
#     for (n, λ) in enumerate(λs)
#         for (i, xout) in enumerate(xs)
#             xsub     = deleteelem(xs,i)
#             ysub     = deleteelem(ys,i)
#             spl      = fit(SmoothingSpline, xsub, ysub, λ)
#             errs[n] += abs(predict(spl, xout)-ys[i])       # compute error at left-out point
#         end
#     end
#     iopt = argmin(errs)
#     λopt = λs[iopt]
#     (iopt == 1 || iopt == L) && @warn "λopt=$λopt is a boundary solution."
#     return λopt
# end
