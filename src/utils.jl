# find root for monotonic univariate functions
function monoroot(f, l::F, u::F; tol = eps(F), maxit = 30, verbose=false) where {F<:AbstractFloat}
    fl = f(l)
    fu = f(u)
    if sign(fl) == sign(fu)     # f monotone & same sign at both ends => no root in interval. still return something to avoid downstream errors
        return u
    end
    h = l
    for _ in 1:maxit
        h  = (l+u)/2
        fh = f(h)
        if abs(fh) < tol
            return h
        elseif sign(fl) == sign(fh)
            l  = h
            fl = fh
        else
            u  = h
            fu = fh
        end
    end
    return h
end

# trapezoidal approximation
# want: c[i] = c(x[i]) = int_0^x_i dx f(x)
# then c[0]=0, and for i>=1 use
# c[i] ≈ sum_{j=1}^i 0.5(f(x_j) + f(x_{j-1}))(x_j - x_{j-1})
# Or recursively,
# c[i] = c[i-1] + 0.5(f(x_i) + f(x_{i-1}))(x_i - x_{i-1})
function trapez!(c::T,xs::T,ys::T) where {K<:AbstractFloat,T<:Vector{K}}
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
function trapez(xs,ys)
    c = similar(ys)
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

# this function computes both forward and backward estimators, reports their weighted mean
const STEPSTONE_FWD_WEIGHT = Ref(0.5)
function stepping_stone!(
    zs::TV,
    bs::TV,
    trVs::Vector{TV}
    ) where {TF<:AbstractFloat, TV<:Vector{TF}}
    w     = STEPSTONE_FWD_WEIGHT[]
    onemw = one(TF) - w
    zs[1] = zero(TF)
    accf  = accb = zero(TF)
    llen  = log(length(trVs[1]))
    for i in 2:length(zs)
        db    = bs[i] - bs[i-1]
        accf += logsumexp(-db*trVs[i-1]) - llen
        llen  = log(length(trVs[i]))
        accb += llen - logsumexp(db*trVs[i]) 
        zs[i] = w*accf + onemw*accb
    end
end
function stepping_stone(bs, trVs)
    zs = similar(bs)
    stepping_stone!(zs, bs, trVs)
    return zs
end
