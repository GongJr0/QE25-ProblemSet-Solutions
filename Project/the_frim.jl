module FirmModel

using LinearAlgebra, Statistics

export Params, solve_model, stationary_distribution,
       compute_moments, compute_aggregates, subsidy_cost, corr_k_z,
       DATA_MOMENTS


Base.@kwdef mutable struct Params
    alpha::Float64 = 0.30
    nu::Float64    = 0.60       # alpha + nu = 0.90 < 1  -> DRS
    delta::Float64 = 0.08
    r::Float64     = 0.04
    w::Float64     = 1.00
    rho::Float64   = 0.90
    sigma::Float64 = 0.12
    zbar::Float64  = exp(-0.12^2 / (2*(1-0.90^2)))   # E[z] = 1
    beta::Float64  = 1/(1+0.04)

    # Adjustment-cost parameters (estimated)
    gamma::Float64 = 1.0
    F::Float64     = 0.02
    ps::Float64    = 0.70
end

const P = Params()

const DATA_MOMENTS = [0.122, 0.081, 0.104, 0.180, 0.014]


function normcdf(x::Float64)
    x < 0.0 && return 1.0 - normcdf(-x)
    t    = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 +
           t * (-0.356563782 +
           t * (1.781477937 +
           t * (-1.821255978 +
           t *  1.330274429))))
    return 1.0 - (1.0 / sqrt(2*pi)) * exp(-0.5 * x * x) * poly
end

#AR(1) tauchen

function tauchen(rho::Float64, sigma::Float64; n::Int=7, m::Int=3)
    sigma_z = sigma / sqrt(1 - rho^2)
    zmax    =  m * sigma_z
    zmin    = -m * sigma_z
    z       = collect(range(zmin, zmax; length=n))
    d       = (zmax - zmin) / (n - 1)

    Pi = zeros(n, n)
    for i in 1:n
        Pi[i, 1] = normcdf((z[1]   - rho*z[i] + d/2) / sigma)
        Pi[i, n] = 1.0 - normcdf((z[n] - rho*z[i] - d/2) / sigma)
        for j in 2:n-1
            Pi[i, j] = normcdf((z[j] - rho*z[i] + d/2) / sigma) -
                       normcdf((z[j] - rho*z[i] - d/2) / sigma)
        end
    end
    return z, Pi
end

#optimal labour (h)
function optimal_labor(z::Float64, k::Float64, p::Params=P)
    return (p.nu * z * k^p.alpha / p.w)^(1/(1 - p.nu))
end
#static profit
function static_profit(z::Float64, k::Float64, p::Params=P)
    h = optimal_labor(z, k, p)
    y = z * k^p.alpha * h^p.nu
    return y - p.w * h
end

#Adjustment cost

function adj_cost_vec(ivec::Vector{Float64}, k::Float64,
                      gamma::Float64, F::Float64)
    quad  = @. gamma/2 * (ivec/k)^2 * k
    fixed = @. ifelse(ivec != 0.0, F * k, 0.0)
    return quad .+ fixed
end

#Price of investment

function price_invest(ivec::Vector{Float64}, ps::Float64, tau::Float64=0.0)
    return @. ifelse(ivec >= 0.0, 1.0 - tau, ps)
end

#Capital Grid

function make_k_grid(p::Params, nk::Int)
    k_ss = ((p.r + p.delta) /
            (p.alpha * (p.nu/p.w)^(p.nu/(1-p.nu))))^(
             1/(p.alpha + p.nu - 1 + p.nu/(1-p.nu)))
    k_ss = max(k_ss, 1.0)
    return collect(range(0.05*k_ss, 4.0*k_ss; length=nk))
end

#VFI

function solve_model(gamma::Float64, F::Float64, ps::Float64;
                     tau::Float64   = 0.0,
                     nk::Int        = 40,
                     nz::Int        = 7,
                     tol::Float64   = 1e-6,
                     maxiter::Int   = 2000,
                     verbose::Bool  = false,
                     p::Params      = P)

    zlog_grid, Pi = tauchen(p.rho, p.sigma; n=nz)
    z_grid = exp.(zlog_grid) .* p.zbar
    k_grid = make_k_grid(p, nk)
    beta   = p.beta

    pi_mat = [static_profit(z_grid[iz], k_grid[ik], p)
              for iz in 1:nz, ik in 1:nk]

    #initial guess
    V = [pi_mat[iz, ik] / (1 - beta) for ik in 1:nk, iz in 1:nz]

    i_star  = zeros(nk, nz)
    ik_star = ones(Int, nk, nz)
    EV      = zeros(nk, nz)
    V_new   = zeros(nk, nz)

    for iter in 1:maxiter
        V_old = copy(V)
        mul!(EV, V_old, Pi')

        fill!(V_new, -Inf)

        for ik in 1:nk
            k        = k_grid[ik]
            i_vec    = k_grid .- (1 - p.delta) * k
            p_vec    = price_invest(i_vec, ps, tau)
            c_vec    = adj_cost_vec(i_vec, k, gamma, F)
            cost_vec = p_vec .* i_vec .+ c_vec

            for iz in 1:nz
                obj_vec = pi_mat[iz, ik] .- cost_vec .+ beta .* EV[:, iz]
                best    = argmax(obj_vec)
                V_new[ik, iz]  = obj_vec[best]
                i_star[ik, iz] = i_vec[best]
                ik_star[ik, iz] = best
            end
        end

        diff = maximum(abs.(V_new .- V_old))
        V .= V_new
        verbose && iter % 100 == 0 &&
            println("  iter $iter  |DV| = $(round(diff; sigdigits=3))")
        if diff < tol
            verbose && println("  Converged at iter $iter, |DV| = $(round(diff; sigdigits=3))")
            break
        end
    end

    return (V=V, i_star=i_star, ik_star=ik_star,
            k_grid=k_grid, z_grid=z_grid, Pi=Pi)
end

#the Young thingy/trick

function stationary_distribution(ik_star::Matrix{Int}, Pi::Matrix{Float64};
                                  tol::Float64=1e-10, maxiter::Int=5000)
    nk, nz = size(ik_star)
    mu     = fill(1.0/(nk*nz), nk, nz)
    mu_new = zeros(nk, nz)

    for _ in 1:maxiter
        fill!(mu_new, 0.0)
        for ik in 1:nk, iz in 1:nz
            ikp = ik_star[ik, iz]
            for izp in 1:nz
                mu_new[ikp, izp] += mu[ik, iz] * Pi[iz, izp]
            end
        end
        diff = maximum(abs.(mu_new .- mu))
        mu .= mu_new
        diff < tol && break
    end
    mu ./= sum(mu)
    return mu
end

#investment rate

function compute_moments(i_star::Matrix{Float64},
                         k_grid::Vector{Float64},
                         mu::Matrix{Float64})
    nk, nz = size(i_star)
    ir = [i_star[ik, iz] / k_grid[ik] for ik in 1:nk, iz in 1:nz]
    r  = vec(ir); w = vec(mu)

    avg_inv   = dot(r, w)
    inaction  = dot(abs.(r) .< 0.01, w)
    neg_inv   = dot(r .< 0.0, w)
    pos_spike = dot(r .>  0.20, w)
    neg_spike = dot(r .< -0.20, w)

    return (avg_inv=avg_inv, inaction=inaction, neg_inv=neg_inv,
            pos_spike=pos_spike, neg_spike=neg_spike)
end

# K,H and Y aggregated

function compute_aggregates(k_grid::Vector{Float64},
                             z_grid::Vector{Float64},
                             mu::Matrix{Float64},
                             p::Params=P)
    nk, nz = size(mu)
    K = H = Y = 0.0
    for ik in 1:nk, iz in 1:nz
        k = k_grid[ik]; z = z_grid[iz]
        h = optimal_labor(z, k, p)
        y = z * k^p.alpha * h^p.nu
        m = mu[ik, iz]
        K += k * m; H += h * m; Y += y * m
    end
    return (K=K, H=H, Y=Y)
end

#cost of subsidy

function subsidy_cost(i_star::Matrix{Float64},
                      mu::Matrix{Float64}, tau::Float64)
    cost = 0.0
    for ik in axes(i_star,1), iz in axes(i_star,2)
        i = i_star[ik, iz]
        i > 0.0 && (cost += tau * i * mu[ik, iz])
    end
    return cost
end

#correlation

function corr_k_z(k_grid::Vector{Float64},
                   z_grid::Vector{Float64},
                   mu::Matrix{Float64})
    nk, nz = size(mu)
    k_vals = [k_grid[ik] for ik in 1:nk for iz in 1:nz]
    z_vals = [z_grid[iz] for ik in 1:nk for iz in 1:nz]
    w      = vec(mu)
    Ek  = dot(k_vals, w); Ez  = dot(z_vals, w)
    Ek2 = dot(k_vals.^2, w); Ez2 = dot(z_vals.^2, w)
    Ekz = dot(k_vals .* z_vals, w)
    vk  = Ek2 - Ek^2; vz = Ez2 - Ez^2
    (vk <= 0 || vz <= 0) && return 0.0
    return (Ekz - Ek*Ez) / sqrt(vk * vz)
end

end # module FirmModel
