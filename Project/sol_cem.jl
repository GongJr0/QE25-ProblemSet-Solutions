using Parameters, QuantEcon, LinearAlgebra, Plots, LaTeXStrings,Statistics, Printf, Optim, PrettyTables, DataFrames
pyplot()

@with_kw struct FirmParams
    α::Float64      = 0.30
    ν::Float64      = 0.60
    δ::Float64      = 0.08
    r::Float64      = 0.04
    w::Float64      = 1.0
    ρ::Float64      = 0.90
    σ::Float64      = 0.12
    Nz::Int         = 11
    Nk::Int         = 250
    θ_grid::Float64 = 0.4
    γ_adj::Float64  = 2.0
    F::Float64      = 0.0
    ps::Float64     = 1.0
end

const γ_LB,  γ_UB  = 0.05,  5.0
const F_LB,  F_UB  = 0.0,   0.05
const ps_LB, ps_UB = 0.30,  0.999

const DATA_MOM = [0.122, 0.081, 0.104, 0.180, 0.014]
const T_IDX    = [2, 4, 5]

const N_FINE_FIXED = 100


# Profit matrix using the FOC given in the question π̃(k,z) = (1-ν)(ν/w)^(ν/(1-ν)) z^(1/(1-ν)) k^(α/(1-ν))

function compute_profit_matrix(k_grid, z_grid, p::FirmParams)
    α, ν, w = p.α, p.ν, p.w
    coeff = (1 - ν) * (ν / w)^(ν / (1 - ν))
    [coeff * z_grid[iz]^(1/(1-ν)) * k_grid[ik]^(α/(1-ν))
     for ik in 1:length(k_grid), iz in 1:length(z_grid)]
end


# Productivity — Rouwenhorst, E[z]=1 normalisation

function setup_productivity(p::FirmParams)
    z_tilde = exp(-p.σ^2 / (2*(1 - p.ρ^2)))
    mc      = QuantEcon.rouwenhorst(p.Nz, p.ρ, p.σ, 0.0)
    exp.(mc.state_values .+ log(z_tilde)), mc.p
end

function find_kss(p::FirmParams)
    A = (1 - p.ν) * (p.ν / p.w)^(p.ν / (1 - p.ν))
    φ = p.α / (1 - p.ν)
    (A * φ / (p.r + p.δ))^(1 / (1 - φ))
end


function capital_grid(p::FirmParams)
    k_ss = find_kss(p)
    k_min = 0.01 * k_ss
    k_max = 25.0 * k_ss
    lg = range(log(k_min), log(k_max), length=p.Nk)
    kg = exp.(lg)
    return kg, k_ss
end


# Investment Variable Cost  price(i)·i + (γ/2)·(i/k)²·k
# Fixed cost F·k handled separately in Bellman (inaction comparison)
function inv_cost(i, k, p::FirmParams)
    price = i >= 0 ? 1.0 : p.ps
    price * i + (p.γ_adj / 2) * (i / k)^2 * k
end

#Linear interpolation for continuation value at non-grid point k_s
function interp(kv, iz, kg, M)
    kv <= kg[1]   && return M[1,   iz]
    kv >= kg[end] && return M[end, iz]
    hi = searchsortedfirst(kg, kv)
    lo = hi - 1
    w  = (kv - kg[lo]) / (kg[hi] - kg[lo])
    (1-w)*M[lo, iz] + w*M[hi, iz]
end

#Bellman operator
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

function bellman_operator!(Vn, pkl, pia, V, kg, zg, Π, profit, p::FirmParams)
    Nk, Nz = size(V)
    β = 1.0 / (1.0 + p.r)
    EV = β .* (V * Π')

    kmin, kmax = kg[1], kg[end]

    for iz in 1:Nz, ik in 1:Nk
        k  = kg[ik]
        π  = profit[ik, iz]
        ks = (1 - p.δ) * k

        # Inaction option (always defined)
        vA = π + interp(ks, iz, kg, EV)

        # Adjust option: choose kp continuously in [kmin,kmax]
        function obj(kp)
            i = kp - ks
            return π - inv_cost(i, k, p) - p.F*k + interp(kp, iz, kg, EV)
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

# Howard Policy Iteration
function howard!(V, pkl, pia, profit, kg, Pi, p::FirmParams; steps=30)
    Nk = length(kg);  Nz = size(Pi,1);  β = 1/(1+p.r)
    for _ in 1:steps
        EV = β .* (V * Pi')
        for iz in 1:Nz, ik in 1:Nk
            k  = kg[ik];  kp = pkl[ik,iz]
            if pia[ik,iz]
                V[ik,iz] = profit[ik,iz] + interp(kp, iz, kg, EV)
            else
                im = clamp(searchsortedfirst(kg, kp), 1, Nk)
                V[ik,iz] = profit[ik,iz] - inv_cost(kp-(1-p.δ)*k,k,p) -
                            p.F*k + EV[im,iz]
            end
        end
    end
end


function vfi_solve(p::FirmParams; tol=1e-6, maxiter=3000,
                   h_every=10, h_steps=30, verbose=true, label="")
    zg, Pi   = setup_productivity(p)
    kg, k_ss = capital_grid(p)
    Nk, Nz   = length(kg), length(zg)
    profit   = compute_profit_matrix(kg, zg, p)

    Vo  = profit ./ p.r
    Vn  = similar(Vo)
    pkl = zeros(Nk, Nz)
    pia = falses(Nk, Nz)

    verbose && @printf("\n  VFI [%s] γ=%.3f F=%.4f ps=%.3f | Nk=%d Nz=%d k_ss=%.3f\n",
                       label, p.γ_adj, p.F, p.ps, Nk, Nz, k_ss)
    t0 = time();  hist = Float64[]

    for iter in 1:maxiter
        bellman_operator!(Vn, pkl, pia, Vo, profit, kg, Pi, p)
        iter % h_every == 0 && howard!(Vn, pkl, pia, profit, kg, Pi, p, steps=h_steps)

        d = maximum(abs.(Vn .- Vo))
        push!(hist, d)

        if d < tol
            el = time() - t0
            verbose && @printf("\nConverged iter=%d  time=%.2fs  sup|ΔV|=%.2e\n",
                               iter, el, d)
            pol_i = [pkl[ik,iz] - (1-p.δ)*kg[ik] for ik in 1:Nk, iz in 1:Nz]
            return (V=Vn, pkl=pkl, pol_i=pol_i, pia=pia,
                    kg=kg, zg=zg, Pi=Pi, profit=profit, k_ss=k_ss,
                    converged=true, niter=iter, elapsed=el, hist=hist)
        end
        Vo .= Vn
    end

    el    = time() - t0
    pol_i = [pkl[ik,iz] - (1-p.δ)*kg[ik] for ik in 1:Nk, iz in 1:Nz]
    verbose && println("\nDid not converge")
    (V=Vn, pkl=pkl, pol_i=pol_i, pia=pia,
     kg=kg, zg=zg, Pi=Pi, profit=profit, k_ss=k_ss,
     converged=false, niter=maxiter, elapsed=el, hist=hist)
end

function stationary_dist(pkl, kg, Pi; tol=1e-10, maxiter=5000, verbose=true)
    Nk = length(kg);  Nz = size(Pi,1)
    mu  = fill(1/(Nk*Nz), Nk, Nz)
    mun = zeros(Nk, Nz)

    for iter in 1:maxiter
        fill!(mun, 0.0)
        for iz in 1:Nz, ik in 1:Nk
            m = mu[ik,iz];  m < 1e-15 && continue
            kp = pkl[ik,iz]
            if     kp <= kg[1];   il,ih,wh = 1,  1,  0.0
            elseif kp >= kg[end]; il,ih,wh = Nk, Nk, 0.0
            else
                ih = searchsortedfirst(kg, kp);  il = ih-1
                wh = (kp - kg[il]) / (kg[ih] - kg[il])
            end
            wl = 1 - wh
            for iz2 in 1:Nz
                q = Pi[iz,iz2]
                mun[il,iz2] += wl*q*m
                ih != il && (mun[ih,iz2] += wh*q*m)
            end
        end
        mun ./= sum(mun)
        if sum(abs.(mun .- mu)) < tol
            verbose && @printf("  μ converged iter=%d  sum=%.10f\n", iter, sum(mun))
            return mun
        end
        mu .= mun
    end
    verbose && println("  WARNING: μ did not converge")
    mu
end

function compute_moments(pol_i, kg, mu)
    ir = pol_i ./ kg
    literal_inaction = pol_i .== 0.0     
    (avg_ir    = sum(ir .* mu),
     inaction  = sum((abs.(ir) .< 0.01) .* mu),
     literal_inaction = sum(literal_inaction .* mu),
     frac_neg  = sum((ir .< 0.0)         .* mu),
     pos_spike = sum((ir .> 0.20)         .* mu),
     neg_spike = sum((ir .< -0.20)        .* mu))
end

function check_boundary_mass(mu, label="")
    lo = sum(mu[1,   :])
    hi = sum(mu[end, :])
    ok = max(lo, hi) < 1e-4
    @printf("  Boundary mass [%s]:  lower=%.2e  upper=%.2e  %s\n",
            label, lo, hi, ok ? "OK" : "GRID TOO NARROW — increase k_max or decrease k_min")
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
        mom = compute_moments(sol.pol_i, sol.kg, μ)

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

function plot_F_ps_heatmaps(df_pct::DataFrame)

    F_vals  = sort(unique(df_pct.F))
    ps_vals = sort(unique(df_pct.ps))

    metrics = [
        (:inaction,         "Inaction (%)"),
        (:literal_inaction, "Literal Inaction (%)"),
        (:avg_ir,           "Avg Investment Rate (%)"),
        (:pos_spike,        "Positive Spike (%)"),
        (:neg_spike,        "Negative Spike (%)")
    ]

    lay = @layout [a b c; d e]

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


function moments_dataframe(mom)
    DataFrame(
        Moment   = ["Avg Investment Rate", "Inaction Rate",
                    "Fraction Negative",   "Positive Spike Rate",
                    "Negative Spike Rate"],
        Value    = round.([mom.avg_ir, mom.inaction, mom.frac_neg,
                    mom.pos_spike, mom.neg_spike] .* 100, digits=3)
    )
end

function show_moments_table(mom, label::String)
    df = DataFrame(
        Moment   = ["Avg Investment Rate (%)", "Inaction Rate (%)",
                    "Fraction Negative (%)",   "Positive Spike Rate (%)",
                    "Negative Spike Rate (%)"],
        Model    = round.([mom.avg_ir, mom.inaction, mom.frac_neg,
                           mom.pos_spike, mom.neg_spike] .* 100, digits=3),
        LRD_Data = [12.2, 8.1, 10.4, 18.0, 1.4])
    rename!(df, :Model => Symbol(label))
    pretty_table(df)
end

function show_sensitivity_table(base, fine, nk_b, nk_f)
    bv = [base.avg_ir, base.inaction, base.frac_neg, base.pos_spike, base.neg_spike] .* 100
    fv = [fine.avg_ir, fine.inaction, fine.frac_neg, fine.pos_spike, fine.neg_spike] .* 100
    ch = abs.(fv .- bv) ./ (abs.(bv) .+ 1e-12) .* 100
    df = DataFrame(Moment   = ["Avg Invest Rate(%)", "Inaction Rate(%)",
                                "Fraction Neg(%)",   "Positive Spike(%)",
                                "Negative Spike(%)"],
                   Baseline = round.(bv, digits=3),
                   Fine     = round.(fv, digits=3),
                   Pct_Chg  = round.(ch, digits=2))
    @printf("\n  Grid Sensitivity: Nk=%d → Nk=%d  (N_FINE_FIXED=%d fixed)\n",
            nk_b, nk_f, N_FINE_FIXED)
    pretty_table(df)
    mx = maximum(ch)
    println(mx < 1.0 ? "\nMax=$(round(mx,digits=3))% < 1% → grid adequate" :
                       "\nMax=$(round(mx,digits=3))% > 1% → increase Nk")
end

function show_cross_table(moms, labels)
    df = DataFrame(
        Moment   = ["Avg Investment Rate(%)", "Inaction Rate(%)",
                    "Fraction Negative(%)",   "Positive Spike Rate(%)",
                    "Negative Spike Rate(%)"],
        LRD_Data = [12.2, 8.1, 10.4, 18.0, 1.4])
    for (m, lb) in zip(moms, labels)
        df[!, Symbol(lb)] = round.([m.avg_ir, m.inaction, m.frac_neg,
                                    m.pos_spike, m.neg_spike] .* 100, digits=2)
    end
    pretty_table(df)
end

# Plots
function z_sel(zg)
    Nz = length(zg)
    idx = [1, div(Nz+1,2), Nz]
    lab = ["Low", "Med", "High"]
    labels = [
        "$(lab[j]) z=$(round(zg[idx[j]], digits=3))"
        for j in eachindex(idx)
    ]

    idx, labels
end

function plot_F_ps_heatmaps(df_pct::DataFrame)

    F_vals  = sort(unique(df_pct.F))
    ps_vals = sort(unique(df_pct.ps))

    metrics = [
        (:inaction,         "Inaction (%)"),
        (:literal_inaction, "Literal Inaction (%)"),
        (:avg_ir,           "Avg Investment Rate (%)"),
        (:pos_spike,        "Positive Spike (%)"),
        (:neg_spike,        "Negative Spike (%)")
    ]

    lay = @layout [a b c; d e]

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

function percent_view(df::DataFrame)
    out = copy(df)
    pct_cols = [:avg_ir, :inaction, :frac_neg, :pos_spike, :neg_spike, :literal_inaction]
    for c in pct_cols
        out[!, c] .= 100 .* out[!, c]
    end
    return out
end

function plot_policy(sol, sfx; save_prefix="")
    kg=sol.kg; V=sol.V; pi=sol.pol_i; kss=sol.k_ss; zg=sol.zg
    zi, zl = z_sel(zg)
    cs = [:royalblue, :firebrick, :forestgreen]
    mkpath(joinpath(@__DIR__,"figures"))

    pV = plot(title="Value Function — $sfx", xlabel=L"k", ylabel=L"V(k,z)",
              legend=:bottomright, size=(820,480))
    pI = plot(title="Investment Rate i/k — $sfx", xlabel=L"k", ylabel=L"i/k",
              legend=:topright, size=(820,480))
    hline!(pI,[0.0],    lw=1.2,c=:black,ls=:dash,label="")
    hline!(pI,[0.2,-0.2],lw=1, c=:grey, ls=:dot, label="±20%")

    for (iz,lb,c) in zip(zi,zl,cs)
        plot!(pV, kg, V[:,iz],          label=lb, lw=2, c=c)
        plot!(pI, kg, pi[:,iz]./kg,     label=lb, lw=2, c=c)
    end
    vline!(pV,[kss], lw=1, c=:black, ls=:dot, label="k_ss")
    vline!(pI,[kss], lw=1, c=:black, ls=:dot, label="k_ss")

    if !isempty(save_prefix)
        savefig(pV, joinpath(@__DIR__,"figures","$(save_prefix)_value.png"))
        savefig(pI, joinpath(@__DIR__,"figures","$(save_prefix)_ir.png"))
    end
    display(pV); display(pI)
end

# Stationary distribution: heatmap + marginal over k + ir histogram
function plot_stationary(sol, mu, sfx; save_prefix="")
    kg=sol.kg; zg=sol.zg; pi=sol.pol_i; kss=sol.k_ss
    mkpath(joinpath(@__DIR__,"figures"))

    # 1. Heatmap μ(k,z)
    ph = heatmap(kg, zg, mu', title="μ(k,z) — $sfx",
                 xlabel=L"k", ylabel=L"z", color=:viridis, size=(820,480))
    vline!(ph,[kss], lw=2, c=:white, ls=:dash, label="k_ss")

    # 2. Marginal over k
    mu_k = vec(sum(mu, dims=2))
    pm = bar(kg, mu_k, title="Marginal μ(k) — $sfx",
             xlabel=L"k", ylabel="Mass", legend=false,
             c=:steelblue, alpha=0.7, size=(820,480))
    vline!(pm,[kss], lw=2, c=:red, ls=:dash, label="k_ss")

    # 3. Weighted histogram of investment rates i/k
    ir_all = vec(pi ./ kg)
    w_all  = vec(mu)
    mask   = w_all .> 1e-12
    ir_p   = ir_all[mask]
    w_p    = w_all[mask] ./ sum(w_all[mask])

    phist = histogram(ir_p, weights=w_p, bins=80,
                      title="Distribution of i/k — $sfx",
                      xlabel=L"i/k", ylabel="Density",
                      legend=false, c=:steelblue, alpha=0.75,
                      xlims=(-0.6,1.0), size=(820,480))
    vline!(phist,[0.0],  lw=2, c=:black, ls=:dash)
    vline!(phist,[0.20], lw=1, c=:red,   ls=:dot)
    vline!(phist,[-0.20],lw=1, c=:red,   ls=:dot)

    if !isempty(save_prefix)
        savefig(ph,    joinpath(@__DIR__,"figures","$(save_prefix)_heatmap.png"))
        savefig(pm,    joinpath(@__DIR__,"figures","$(save_prefix)_marginal_k.png"))
        savefig(phist, joinpath(@__DIR__,"figures","$(save_prefix)_ir_hist.png"))
    end
    display(ph); display(pm); display(phist)
end


function to_phys(θ)
    sig(x) = 1 / (1 + exp(-x))
    γ  = γ_LB  + (γ_UB  - γ_LB)  * sig(θ[1])
    F  = F_LB  + (F_UB  - F_LB)  * sig(θ[2])
    ps = ps_LB + (ps_UB - ps_LB) * sig(θ[3])
    γ, F, ps
end

function to_raw(γ, F, ps)
    logit(p,lo,hi) = log((p-lo+1e-8)/(hi-p+1e-8))
    [logit(γ,γ_LB,γ_UB), logit(F,F_LB,F_UB), logit(ps,ps_LB,ps_UB)]
end

function model_mom_vec(γ, F, ps; Nk=200, Nz=11)
    p   = FirmParams(γ_adj=γ, F=F, ps=ps, Nk=Nk, Nz=Nz)
    sol = vfi_solve(p, verbose=false)
    mu  = stationary_dist(sol.pkl, sol.kg, sol.Pi, verbose=false)
    m   = compute_moments(sol.pol_i, sol.kg, mu)
    [m.avg_ir, m.inaction, m.frac_neg, m.pos_spike, m.neg_spike]
end

function smm_loss(θ)
    γ,F,ps  = to_phys(θ)
    mv      = model_mom_vec(γ, F, ps)
    md      = DATA_MOM[T_IDX]
    sum(((mv[T_IDX] .- md) ./ (md .+ 1e-12)).^2)
end

function grid_search_smm(; verbose=true)
    γ_grid  = [0.3, 0.7, 1.2, 2.0]
    F_grid  = [0.001, 0.005, 0.01, 0.02]
    ps_grid = [0.5, 0.7, 0.85, 0.95]

    best_loss = Inf
    best_γ, best_F, best_ps = 1.0, 0.005, 0.7

    verbose && println("\n  Grid search over $(length(γ_grid)*length(F_grid)*length(ps_grid)) points...")
    for γ in γ_grid, F in F_grid, ps in ps_grid
        try
            loss = smm_loss(to_raw(γ, F, ps))
            if loss < best_loss
                best_loss = loss
                best_γ, best_F, best_ps = γ, F, ps
                verbose && @printf("  New best: γ=%.3f F=%.4f ps=%.3f → loss=%.4f\n",
                                   γ, F, ps, loss)
            end
        catch
            # Skip parameter combinations that cause numerical issues
        end
    end
    verbose && @printf("\n  Grid search best: γ=%.3f F=%.4f ps=%.3f  loss=%.4f\n",
                       best_γ, best_F, best_ps, best_loss)
    best_γ, best_F, best_ps
end

function run_smm(; verbose=true)
    # Step 1: grid search for starting values
    γ0, F0, ps0 = grid_search_smm(verbose=verbose)

    # Step 2: Nelder-Mead from best grid point
    verbose && @printf("\n  Nelder-Mead from γ=%.3f F=%.4f ps=%.3f\n", γ0, F0, ps0)
    θ0  = to_raw(γ0, F0, ps0)
    res = optimize(smm_loss, θ0, NelderMead(),
                   Optim.Options(iterations=500, x_abstol=1e-5,
                                 f_reltol=1e-7, show_trace=verbose,
                                 show_every=50))

    γh, Fh, psh = to_phys(res.minimizer)
    if verbose
        println("\n  $(Optim.converged(res) ? "CONVERGED" : "Not converged")")
        @printf("  loss=%.6f\n  γ̂=%.4f  F̂=%.5f  p̂s=%.4f\n", res.minimum, γh, Fh, psh)
    end
    γh, Fh, psh, res
end

#Main execution


# STAGE 1 — Convex only  (F=0, ps=1, γ=2)
# Expected: smooth policy, NO inaction, continuous investment rate

println("\n","█"^60,"\n  STAGE 1 — Convex Only (F=0, ps=1, γ=2)\n","█"^60)
p1  = FirmParams(γ_adj=2.0, F=0.0, ps=1.0, Nk=250, Nz=11)
s1  = vfi_solve(p1, verbose=true, label="S1")
mu1 = stationary_dist(s1.pkl, s1.kg, s1.Pi, verbose=true)
m1  = compute_moments(s1.pol_i, s1.kg, mu1)
check_boundary_mass(mu1, "S1")
println("  No inaction (F=0): ", all(.!s1.pia) ? "expected" : "unexpected")
show_moments_table(m1, "Stage 1")
plot_policy(s1, "Convex Only", save_prefix="s1")
plot_stationary(s1, mu1, "Convex Only", save_prefix="s1")


# STAGE 2 — Fixed costs  (F=0.05, ps=1, γ=2)

println("\nFixed Costs (F=0.05, ps=1, γ=2)\n")
p2  = FirmParams(γ_adj=2.0, F=0.05, ps=1.0, Nk=250, Nz=11)
s2  = vfi_solve(p2, verbose=true, label="S2")
mu2 = stationary_dist(s2.pkl, s2.kg, s2.Pi, verbose=true)
m2  = compute_moments(s2.pol_i, s2.kg, mu2)
check_boundary_mass(mu2, "S2")
println("  Inaction: S1=$(round(m1.inaction*100,digits=1))% → S2=$(round(m2.inaction*100,digits=1))%")
show_moments_table(m2, "Stage 2")
plot_policy(s2, "Fixed Costs F=0.05", save_prefix="s2")
plot_stationary(s2, mu2, "Fixed Costs", save_prefix="s2")



# STAGE 3 — Irreversibility  (F=0.05, ps=0.5, γ=2)

println("\nSTAGE 3 — Irreversibility (F=0.05, ps=0.5, γ=2)\n")
p3  = FirmParams(γ_adj=2.0, F=0.05, ps=0.
, Nk=250, Nz=11)
s3  = vfi_solve(p3, verbose=true, label="S3")
mu3 = stationary_dist(s3.pkl, s3.kg, s3.Pi, verbose=true)
m3  = compute_moments(s3.pol_i, s3.kg, mu3)
check_boundary_mass(mu3, "S3")
println("  Frac neg:  S2=$(round(m2.frac_neg*100,digits=2))% → S3=$(round(m3.frac_neg*100,digits=2))%")
println("  Neg spike: S2=$(round(m2.neg_spike*100,digits=2))% → S3=$(round(m3.neg_spike*100,digits=2))%")
show_moments_table(m3, "Stage 3")
plot_policy(s3, "Irreversibility F=0.05 ps=0.5", save_prefix="s3")
plot_stationary(s3, mu3, "Irreversibility", save_prefix="s3")



# STAGE 4 — SMM Calibration
println("\nSTAGE 4 — SMM CALIBRATION\n")
γh, Fh, psh, smm_res = run_smm(verbose=true)

println("\n  Re-solving on fine grid (Nk=350, Nz=13)...")
pc   = FirmParams(γ_adj=γh, F=Fh, ps=psh, Nk=350, Nz=13)
sc   = vfi_solve(pc, verbose=true, label="Calibrated")
muc  = stationary_dist(sc.pkl, sc.kg, sc.Pi, verbose=true)
mc   = compute_moments(sc.pol_i, sc.kg, muc)
check_boundary_mass(muc, "Calibrated")

@printf("\n  CALIBRATED:  γ̂=%.4f  F̂=%.5f  p̂s=%.4f\n", γh, Fh, psh)
show_moments_table(mc, "Calibrated")
plot_policy(sc, "Calibrated", save_prefix="cal")
plot_stationary(sc, muc, "Calibrated", save_prefix="cal")


# Cross Stage Summary
println("\nCROSS-STAGE MOMENT SUMMARY\n")
show_cross_table([m1,m2,m3,mc], ("Stage1","Stage2","Stage3","Calibrated"))
@printf("\n  γ̂=%.4f  F̂=%.5f  p̂s=%.4f\n", γh, Fh, psh)
