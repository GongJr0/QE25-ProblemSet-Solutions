using Parameters, QuantEcon, LinearAlgebra, Plots, Measures, LaTeXStrings, Statistics, Printf, Optim, PrettyTables, DataFrames, Random
gr()


@with_kw struct Params
    α::Float64      = 0.30
    v::Float64      = 0.60
    δ::Float64      = 0.08
    r::Float64      = 0.04
    w::Float64      = 1.0
    ρ::Float64      = 0.90
    σ::Float64      = 0.12
    Nz::Int         = 11
    Nk::Int         = 250
    θ_grid::Float64 = 0.4  # Redundant with log grid, adjust if using power grid.
    γ_adj::Float64  = 2.0
    F::Float64      = 0.0
    ps::Float64     = 1.0
end

const ALL_MOM = (
    avg_ir    = 0.122,
    inaction  = 0.081,
    frac_neg  = 0.104,
    pos_spike = 0.180,
    neg_spike = 0.014
)

const TARGET_MOMENTS = (:avg_ir, :inaction, :pos_spike)

moment_vector(m) = Float64[getfield(m, mom) for mom in TARGET_MOMENTS]

const DATA_VEC = moment_vector(ALL_MOM)

# Optimal labor h*(k,z)
h_star(k, z, p) = ((p.v * z * k^p.α) / p.w)^(1 / (1 - p.v))

# Reduced-form profit π̃(k,z)
pi_tilde(k, z, p) = (1 - p.v) * (p.v / p.w)^(p.v / (1 - p.v)) * z^(1 / (1 - p.v)) * k^(p.α / (1 - p.v))

# Investment price p(i)
price(i, p) = i >= 0 ? 1.0 : p.ps

# Variable adjustment cost
var_adj_cost(i, k, p) = price(i,p) * i + (p.γ_adj/2) * (i/k)^2 * k

# Period payoff (profit minus costs);
payoff(k, z, i, p; adjust::Bool) = pi_tilde(k,z,p) - var_adj_cost(i,k,p) - (adjust ? p.F*k : 0.0)


function find_kss(p::Params)
    A = (1 - p.v) * (p.v / p.w)^(p.v / (1 - p.v))
    ϕ = p.α / (1 - p.v)
    (A * ϕ / (p.r + p.δ))^(1 / (1 - ϕ))
end

function capital_grid(p::Params)
    k_ss = find_kss(p)
    k_min = 0.01 * k_ss
    k_max = 25.0 * k_ss
    lg = range(log(k_min), log(k_max), length=p.Nk)
    kg = exp.(lg)
    return kg, k_ss, k_min, k_max
end

function productivity_grid(p::Params)
    mc = rouwenhorst(p.Nz, p.ρ, p.σ, 0.0)
    Π = mc.p
    log_z = mc.state_values
    z = exp.(log_z)

    # Dist
    vals, vector = eigen(Π')
    i = argmax(real(vals))
    π = real(vector[:, i])
    π ./= sum(π)

    # Normalize
    z ./= sum(π .* z)
    return z, Π, π
end

function k_interp(kp, iz, kg, M)
    if kp <= kg[1]
        return M[1, iz]
    elseif kp >= kg[end]
        return M[end, iz]
    else
        hi = searchsortedfirst(kg, kp)
        lo = hi-1
        w = (kp-kg[lo]) / (kg[hi]-kg[lo]) # Normalized distance of kp between kg[lo] and kg[hi]
        return (1-w)*M[lo, iz] + w*M[hi, iz]
    end
end

function policy_interp(k, iz, kg, pkl)
    if k <= kg[1]
        return pkl[1, iz]
    elseif k >= kg[end]
        return pkl[end, iz]
    else
        hi = searchsortedfirst(kg, k)
        lo = hi - 1
        w = (k - kg[lo]) / (kg[hi] - kg[lo])
        return (1-w) * pkl[lo, iz] + w * pkl[hi, iz]
    end
end

# Golden-section search for maximum on [a,b]
function golden_max(f, a, b; tol=1e-8, maxiter=200)
    ϕ = (sqrt(5) - 1) / 2  # ~0.618
    c = b - ϕ*(b-a)
    d = a + ϕ*(b-a)
    fc = f(c)
    fd = f(d)

    for _ in 1:maxiter
        if (b - a) < tol
            break
        end
        if fc > fd
            b, d, fd = d, c, fc
            c = b - ϕ*(b-a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + ϕ*(b-a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

function bellman_op!(Vn, pkl, pia, V, kg, zg, Π, profit, p::Params)
    Nk, Nz = size(V)
    β = 1.0 / (1.0 + p.r)
    EV = β .* (V * Π')

    kmin, kmax = kg[1], kg[end]

    for iz in 1:Nz, ik in 1:Nk
        k  = kg[ik]
        π  = profit[ik, iz]
        ks = (1 - p.δ) * k

        # Inaction option (always defined)
        vA = π + k_interp(ks, iz, kg, EV)

        # Adjust option: choose kp continuously in [kmin,kmax]
        function obj(kp)
            i = kp - ks
            return π - var_adj_cost(i, k, p) - p.F*k + k_interp(kp, iz, kg, EV)
        end

        kp_star, vB = golden_max(obj, kmin, kmax; tol=1e-8, maxiter=200)

        if vA >= vB
            Vn[ik, iz] = vA
            pkl[ik, iz] = ks
            pia[ik, iz] = true
        else
            Vn[ik, iz] = vB
            pkl[ik, iz] = kp_star
            pia[ik, iz] = false
        end
    end
end

function howard!(V, pkl, pia, profit, kg, Π, p::Params; steps=20)
    Nk, Nz = size(V)
    β = 1.0 / (1.0 + p.r)

    for _ in 1:steps
        EV = β .* (V * Π')

        @inbounds for iz in 1:Nz, ik in 1:Nk
            k  = kg[ik]
            kp = pkl[ik, iz]
            π  = profit[ik, iz]
            ks = (1.0 - p.δ) * k

            if pia[ik, iz]
                # inaction branch: i = 0, no fixed cost
                V[ik, iz] = π + k_interp(kp, iz, kg, EV)
            else
                # adjust branch: i = kp - ks, pay fixed cost
                i = kp - ks
                V[ik, iz] = π - var_adj_cost(i, k, p) - p.F * k + k_interp(kp, iz, kg, EV)
            end
        end
    end
end

function vfi_solve(p::Params; atol=1e-6, maxiter=2000, h_every=5, h_steps=20, verbose=true)
    zg, Π, π = productivity_grid(p)
    kg, k_ss, k_min, k_max = capital_grid(p)
    Nk, Nz = length(kg), length(zg)

    profit = [pi_tilde(kg[ik], zg[iz], p) for ik in 1:Nk, iz in 1:Nz]
    V = profit ./ p.r
    Vn = similar(V)
    pkl = zeros(Nk, Nz)
    pia = falses(Nk, Nz)
    d = Inf

    for it in 1:maxiter
        bellman_op!(Vn, pkl, pia, V, kg, zg, Π, profit, p)
        if h_every > 0 && (it % h_every == 0)
            howard!(Vn, pkl, pia, profit, kg, Π, p; steps=h_steps)
        end

        d = maximum(abs.(Vn .- V))
        V .= Vn
        if verbose && (it % 50 == 0)
            @printf("Iteration %d: max|Vn-V| = %.6f\n", it, d)
        end
        if d < atol
            verbose && @printf("converged iter=%d sup|ΔV|=%.3e\n", it, d)
            return (V=V, pkl=pkl, pia=pia, kg=kg, zg=zg, Π=Π, profit=profit, k_ss=k_ss)
        end
    end
    error("VFI did not converge after $maxiter iterations. Final sup|ΔV| = $d")
end

function stationary_dist(pkl, kg, Π; tol=1e-12, maxiter=50_000, verbose=true)
    Nk = length(kg)
    Nz = size(Π, 1)
    @assert size(pkl) == (Nk, Nz)
    @assert size(Π, 2) == Nz

    μ  = fill(1.0 / (Nk * Nz), Nk, Nz)
    μn = zeros(Nk, Nz)

    for it in 1:maxiter
        fill!(μn, 0.0)

        for iz in 1:Nz, ik in 1:Nk
            m = μ[ik, iz]
            m < 1e-18 && continue

            kp = pkl[ik, iz]

            # locate kp in kg and compute (il, ih, wl, wh)
            if kp <= kg[1]
                il = 1; ih = 1; wl = 1.0; wh = 0.0
            elseif kp >= kg[end]
                il = Nk; ih = Nk; wl = 1.0; wh = 0.0
            else
                ih = searchsortedfirst(kg, kp)
                il = ih - 1
                wh = (kp - kg[il]) / (kg[ih] - kg[il])
                wl = 1.0 - wh
            end

            # move mass across z using Π[iz, iz2]
            @inbounds for iz2 in 1:Nz
                q = Π[iz, iz2]
                μn[il, iz2] += wl * q * m
                if ih != il
                    μn[ih, iz2] += wh * q * m
                end
            end
        end

        s = sum(μn)
        μn ./= s

        dist = sum(abs.(μn .- μ))
        if dist < tol
            verbose && @printf("μ converged iter=%d  L1=%.3e  sum=%.10f\n", it, dist, sum(μn))
            return μn
        end
        μ .= μn
    end

    verbose && @printf("WARNING: μ did not converge after %d iters (last L1=%.3e)\n", maxiter, sum(abs.(μn .- μ)))
    return μ
end

function boundary_mass(μ)
    lo = sum(μ[1, :])
    hi = sum(μ[end, :])
    return lo, hi
end

function compute_moments(pkl, kg, μ, p::Params)
    Nk = length(kg)
    Nz = size(μ, 2)
    @assert size(pkl) == (Nk, Nz)

    ir = zeros(Nk, Nz)
    @inbounds for iz in 1:Nz, ik in 1:Nk
        k  = kg[ik]
        ks = (1 - p.δ) * k
        i  = pkl[ik, iz] - ks
        ir[ik, iz] = i / k
    end

    avg_ir    = sum(ir .* μ)
    inaction  = sum((abs.(ir) .< 0.01) .* μ)
    frac_neg  = sum((ir .< 0.0) .* μ)
    pos_spike = sum((ir .> 0.20) .* μ)
    neg_spike = sum((ir .< -0.20) .* μ)

    return (
        avg_ir=avg_ir, 
        inaction=inaction, 
        frac_neg=frac_neg, 
        pos_spike=pos_spike, 
        neg_spike=neg_spike,
        )

end

function moments_df(mom)
    head = ["Avg Investment Rate", "Inaction", "Fraction Negative", "Positive Spike", "Negative Spike"]
    data = [getfield(mom, :avg_ir)*100, getfield(mom, :inaction)*100, getfield(mom, :frac_neg)*100,
            getfield(mom, :pos_spike)*100, getfield(mom, :neg_spike)*100]
    df = DataFrame(Metric=head, Value=data)
    return df
end

function sweep_F_ps(; 
    F_grid = [0.0, 0.0001, 0.0002, 0.00035, 0.0005],
    ps_grid = [1.0, 0.95, 0.85, 0.70, 0.50],
    verbose = false
)
    # Very Slow Function, try to compute once and store.

    rows = NamedTuple[]

    for F in F_grid, ps in ps_grid
        p = Params(F=F, ps=ps)

        sol = vfi_solve(p, verbose=verbose)
        μ   = stationary_dist(sol.pkl, sol.kg, sol.Π, verbose=verbose)
        mom = compute_moments(sol.pkl, sol.kg, μ, p)

        sha_lo = mean(sol.pkl .<= sol.kg[1]  + 1e-8)
        sha_hi = mean(sol.pkl .>= sol.kg[end] - 1e-8)

        lo_mass, hi_mass = boundary_mass(μ)

        literal_inaction = sum(sol.pia .* μ)

        push!(rows, (
            F = F,
            ps = ps,

            sha_kmin = sha_lo,
            sha_kmax = sha_hi,

            mass_kmin = lo_mass,
            mass_kmax = hi_mass,

            avg_ir = mom.avg_ir,
            inaction = mom.inaction,
            frac_neg = mom.frac_neg,
            pos_spike = mom.pos_spike,
            neg_spike = mom.neg_spike,

            literal_inaction = literal_inaction
        ))
    end
    df = DataFrame(rows)
    sort!(df, [:F, :ps])
    return df
end

function percent_view(df::DataFrame)
    out = copy(df)
    pct_cols = [:sha_kmin, :sha_kmax, :mass_kmin, :mass_kmax,
                :avg_ir, :inaction, :frac_neg, :pos_spike, :neg_spike,
                :literal_inaction]
    for c in pct_cols
        out[!, c] .= 100 .* out[!, c]
    end
    return out
end

function plot_F_ps_heatmaps(df_pct::DataFrame)

    F_vals  = sort(unique(df_pct.F))
    ps_vals = sort(unique(df_pct.ps))

    metrics = [
        (:inaction,         "Inaction (%)"),
        (:literal_inaction, "Literal Inaction (%)"),
        (:avg_ir,           "Avg Investment Rate (%)"),
        (:frac_neg,         "Fraction Negative (%)"),
        (:pos_spike,        "Positive Spike (%)"),
        (:neg_spike,        "Negative Spike (%)")
    ]

    lay = @layout [a b c; d e f]

    plt = plot(layout=lay, size=(1200, 1000), margin=5mm)

    for (i, (col, ttl)) in enumerate(metrics)
        Z = [df_pct[(df_pct.F .== F) .& (df_pct.ps .== ps), col][1]
             for ps in ps_vals, F in F_vals]

        heatmap!(
            plt[i],
            F_vals,
            ps_vals,
            Z,
            xlabel=L"F",
            ylabel=L"p_s",
            xrotation=45,
            xformatter=:scientific,
            title=ttl,
            color=:viridis,
            colorbar=true
        )
    end
    return plt
end

function plot_stationary(sol, mu, p, sfx; save_prefix="")
    kg = sol.kg
    kp = sol.pkl
    zg = sol.zg
    mkpath(joinpath(@__DIR__,"figures"))

    # 1. Heatmap μ(k,z)
    ph = heatmap(kg, zg, mu', title="μ(k,z) — $sfx",
                 xlabel=L"k", ylabel=L"z", color=:viridis, size=(820,480))

    # 2. Marginal over k
    mu_k = vec(sum(mu, dims=2))
    pm = bar(kg, mu_k, title="Marginal μ(k) — $sfx",
             xlabel=L"k", ylabel="Mass", legend=false,
             c=:steelblue, alpha=0.7, size=(820,480))

    # 3. Weighted histogram of investment rates i/k
    ir_all = vec((kp .- (1 - p.δ) .* kg) ./ kg)
    w_all  = vec(mu)
    mask   = w_all .> 1e-12
    ir_p   = ir_all[mask]
    w_p    = w_all[mask] ./ sum(w_all[mask])

    phist = histogram(ir_p, weights=w_p, bins=80,
                      title="Distribution of i/k — $sfx",
                      xlabel=L"i/k", ylabel="Density",
                      legend=false, c=:steelblue, alpha=0.75,
                      size=(820,480))

    if !isempty(save_prefix)
        savefig(ph,    joinpath(@__DIR__,"figures","$(save_prefix)_heatmap.png"))
        savefig(pm,    joinpath(@__DIR__,"figures","$(save_prefix)_marginal_k.png"))
        savefig(phist, joinpath(@__DIR__,"figures","$(save_prefix)_ir_hist.png"))
    end
    display(ph); display(pm); display(phist)
end


# S1 (F=0.0, ps=1.0)
p = Params(F=0.0, ps=1.0)
sol_s1 = vfi_solve(p, verbose=true)
mu_s1 = stationary_dist(sol_s1.pkl, sol_s1.kg, sol_s1.Π, verbose=true)
mom_s1 = compute_moments(sol_s1.pkl, sol_s1.kg, mu_s1, p) |> moments_df
println("Moments for S1 (F=0.0, ps=1.0):")
pretty_table(mom_s1)

plot_stationary(sol_s1, mu_s1, p, "S1", save_prefix="s1")

F_grid = [0.0, 0.0001, 0.0002, 0.00035, 0.0005, 0.00075, 0.001, 0.002, 0.003, 0.005]
ps_grid = [1.0, 0.95, 0.85, 0.70, 0.50]

df_sweep = sweep_F_ps(F_grid=F_grid, ps_grid=ps_grid, verbose=true)
df_sweep_pct = percent_view(df_sweep)
plot_F_ps_heatmaps(df_sweep_pct)



sha_hi = mean(sol_s1.pkl .>= sol_s1.kg[end] - 1e-8) 
sha_lo = mean(sol_s1.pkl .<= sol_s1.kg[1] + 1e-8) 
@printf("F=%.4f, ps=%.4f, SHA at k_min=%.3f, SHA at k_max=%.3f\n", p.F, p.ps, sha_lo, sha_hi) 

lo_mass, hi_mass = boundary_mass(mu_s1) 
@printf("F=%.4f, ps=%.4f, mass at k_min=%.3e, mass at k_max=%.3e\n", p.F, p.ps, lo_mass, hi_mass) 


literal_inaction = sum(sol_s1.pia .* mu_s1) 
@printf("F=%.4f, ps=%.4f, literal inaction=%.3e\n", p.F, p.ps, literal_inaction)