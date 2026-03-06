######################
#We got continum of firms with indexes by j from [0,1].
#firms choose the labour they gonna use (h) and how much they will invest (i)
#AR(1) is discretized using tauchen
#capital grid consists of 45 points
#uses NelderMead from optim






using Printf, Statistics, LinearAlgebra, Optim, Plots, Plots.PlotMeasures

gr()
include("the_firm.jl")
using .FirmModel

mkpath("figures")


function run_pipeline(γ, F, ps; tau=0.0, nk=40, nz=7, label="")
    @printf("\n%s\n  Solving: γ=%.3f F=%.4f ps=%.3f tau=%.2f  %s\n",
            "─"^55, γ, F, ps, tau, label)
    t0  = time()
    sol = solve_model(Float64(γ), Float64(F), Float64(ps);
                      tau=Float64(tau), nk=nk, nz=nz)
    μ   = stationary_distribution(sol.ik_star, sol.Pi)
    mom = compute_moments(sol.i_star, sol.k_grid, μ)
    agg = compute_aggregates(sol.k_grid, sol.z_grid, μ)
    sc  = subsidy_cost(sol.i_star, μ, Float64(tau))
    cr  = corr_k_z(sol.k_grid, sol.z_grid, μ)
    @printf("  Done in %.1fs  |  avg_i/k=%.3f  inaction=%.3f\n",
            time()-t0, mom.avg_inv, mom.inaction)
    return merge(sol, (μ=μ, moments=mom, K=agg.K, H=agg.H, Y=agg.Y,
                       cost=sc, corr=cr))
end

inv_rate_mat(i_star, k_grid) =
    [i_star[ik,iz] / k_grid[ik] for ik in axes(i_star,1), iz in axes(i_star,2)]

COLORS = [:steelblue, :firebrick, :darkgreen, :purple, :darkorange]


res_convex = run_pipeline(0.5, 0.00, 1.0;  label="convex only")
res_fixed  = run_pipeline(0.5, 0.03, 1.0;  label="+ fixed costs")
res_irrev  = run_pipeline(0.5, 0.03, 0.70; label="+ irreversibility")

# ── Figure 1: Exploration policy functions ───────────────────
let
    results = [res_convex, res_fixed, res_irrev]
    titles  = ["(a) Convex only\n(γ=0.5, F=0, pₛ=1)",
               "(b) + Fixed costs\n(γ=0.5, F=0.03, pₛ=1)",
               "(c) + Irreversibility\n(γ=0.5, F=0.03, pₛ=0.7)"]

    plts = map(zip(results, titles)) do (res, ttl)
        kg   = res.k_grid
        nz_  = length(res.z_grid)
        iz_p = [1, nz_÷2+1, nz_]
        ir   = inv_rate_mat(res.i_star, kg) .* 100
        p = plot(; xlabel="Capital k", ylabel="i/k (%)", title=ttl,
                   titlefontsize=9, legend=:topright)
        for (ci, iz) in enumerate(iz_p)
            plot!(p, kg, ir[:,iz]; color=COLORS[ci], lw=2,
                  label="z=$(round(res.z_grid[iz]; digits=2))")
        end
        hline!(p, [0]; color=:black, lw=0.8, ls=:dash, label="")
        p
    end

    fig1 = plot(plts...; layout=(1,3), size=(1300,380),
                plot_title="Step 2: Exploration – Policy Functions i*(k,z)",
                left_margin=5mm, bottom_margin=5mm)
    savefig(fig1, "figures/fig1_exploration_policy.png")
    println("Saved figures/fig1_exploration_policy.png")
end


#SMM Calibration


println("\n" * "═"^55)
println("  STEP 3: SMM CALIBRATION")
println("═"^55)

function model_moments_vec(θ; nk=30, nz=7)
    γ, F, ps = θ
    (γ < 0.05 || F < 0 || ps < 0.05 || ps > 1.0) && return fill(999.0, 5)
    try
        sol = solve_model(γ, F, ps; tau=0.0, nk=nk, nz=nz)
        μ   = stationary_distribution(sol.ik_star, sol.Pi)
        m   = compute_moments(sol.i_star, sol.k_grid, μ)
        return [m.avg_inv, m.inaction, m.neg_inv, m.pos_spike, m.neg_spike]
    catch
        return fill(999.0, 5)
    end
end

W_smm = Diagonal([1.0, 2.0, 1.0, 1.0, 4.0])

smm_obj(θ) = begin
    mv   = model_moments_vec(θ)
    diff = mv .- DATA_MOMENTS
    dot(diff, W_smm * diff)
end

# Grid search
println("\n  Grid search over (γ, F, ps) …")
best_obj   = Inf
best_theta = [1.5, 0.02, 0.60]

for γ  in [0.3, 0.6, 1.0, 1.5],
    F  in [0.005, 0.02, 0.05],
    ps in [0.60, 0.80, 0.95]

    obj = smm_obj([γ, F, ps])
    if obj < best_obj
        best_obj   = obj
        best_theta = [γ, F, ps]
        mv = model_moments_vec([γ, F, ps])
        @printf("    γ=%.2f F=%.3f ps=%.2f  obj=%.4f  avg=%.3f inact=%.3f neg=%.3f pos_spk=%.3f neg_spk=%.3f\n",
                γ, F, ps, obj, mv[1], mv[2], mv[3], mv[4], mv[5])
    end
end

@printf("\n  Best from grid: γ=%.3f F=%.4f ps=%.3f\n", best_theta...)

# ── Local refinement with Optim.jl NelderMead ────────────────
println("  Local refinement (Optim.jl NelderMead) …")

opt_result = optimize(
    smm_obj,
    best_theta,
    NelderMead(),
    Optim.Options(
        x_abstol   = 0.01,
        f_abstol   = 1e-3,
        iterations = 80,
        show_trace = false,
    )
)

θ_hat  = Optim.minimizer(opt_result)
γ_hat  = clamp(θ_hat[1], 0.05, Inf)
F_hat  = max(θ_hat[2], 0.0)
ps_hat = clamp(θ_hat[3], 0.05, 0.99)

@printf("  Refined: γ=%.4f  F=%.5f  ps=%.4f\n", γ_hat, F_hat, ps_hat)
@printf("  Optim converged: %s  (f=%.6f)\n",
        Optim.converged(opt_result), Optim.minimum(opt_result))

# Final calibrated baseline
res_base = run_pipeline(γ_hat, F_hat, ps_hat; nk=45, nz=7, label="CALIBRATED BASELINE")
m_hat    = [res_base.moments.avg_inv, res_base.moments.inaction,
            res_base.moments.neg_inv, res_base.moments.pos_spike,
            res_base.moments.neg_spike]

println("\n  Moment fit:")
mom_labels = ["Avg i/k      ", "Inaction     ", "Neg Inv      ",
              "Pos Spike    ", "Neg Spike    "]
for (l, d, m) in zip(mom_labels, DATA_MOMENTS, m_hat)
    @printf("    %s  data=%5.1f%%  model=%5.1f%%\n", l, 100d, 100m)
end

# ── Figure 2: Calibrated V and i* ────────────────────────────
let
    kg     = res_base.k_grid
    nz_    = length(res_base.z_grid)
    iz_plt = [1, nz_÷2+1, nz_]
    ir     = inv_rate_mat(res_base.i_star, kg) .* 100

    pV = plot(; xlabel="Capital k", ylabel="V(k,z)",
                title="Value Function V(k, z)", legend=:bottomright)
    pi = plot(; xlabel="Capital k", ylabel="i*/k (%)",
                title="Policy Function i*(k, z)", legend=:topright)

    for (ci, iz) in enumerate(iz_plt)
        lbl = "z=$(round(res_base.z_grid[iz]; digits=2))"
        plot!(pV, kg, res_base.V[:,iz]; color=COLORS[ci], lw=2, label=lbl)
        plot!(pi, kg, ir[:,iz];         color=COLORS[ci], lw=2, label=lbl)
    end
    hline!(pi, [0];     color=:black, lw=0.8, ls=:dash, label="")
    hline!(pi, [1, -1]; color=:grey,  lw=0.6, ls=:dot,  label="")

    fig2 = plot(pV, pi; layout=(1,2), size=(1100,420),
                plot_title="Calibrated Model  (γ=$(round(γ_hat;digits=3)), " *
                           "F=$(round(F_hat;digits=4)), pₛ=$(round(ps_hat;digits=3)))",
                left_margin=5mm, bottom_margin=5mm)
    savefig(fig2, "figures/fig2_calibrated_model.png")
    println("Saved figures/fig2_calibrated_model.png")
end

# ── Figure 3: Moment fit bar chart ───────────────────────────
let
    xlabels = ["Avg i/k", "Inaction |i/k|<1%", "Neg Inv",
               "Pos Spike i/k>20%", "Neg Spike i/k<-20%"]
    xs = 1:5
    w  = 0.3
    fig3 = bar(xs .- w/2, DATA_MOMENTS .* 100;
               bar_width=w, color=COLORS[1], alpha=0.85, label="Data (LRD)",
               xticks=(xs, xlabels), ylabel="Percent (%)",
               title="SMM Moment Fit: Data vs Model",
               size=(800, 420), left_margin=5mm, bottom_margin=8mm,
               legend=:topright)
    bar!(fig3, xs .+ w/2, m_hat .* 100;
         bar_width=w, color=COLORS[2], alpha=0.85, label="Model")
    savefig(fig3, "figures/fig3_moment_fit.png")
    println("Saved figures/fig3_moment_fit.png")
end

#Policy analysis, tau = 0.10


println("\n" * "═"^55)
println("  STEP 4: INVESTMENT SUBSIDY (tau = 0.10)")
println("═"^55)

res_sub = run_pipeline(γ_hat, F_hat, ps_hat; tau=0.10, nk=45, nz=7, label="SUBSIDY tau=0.10")

m_sub = [res_sub.moments.avg_inv, res_sub.moments.inaction,
         res_sub.moments.neg_inv, res_sub.moments.pos_spike,
         res_sub.moments.neg_spike]

Kb, Hb, Yb = res_base.K, res_base.H, res_base.Y
Ks, Hs, Ys = res_sub.K,  res_sub.H,  res_sub.Y
cost_frac   = res_sub.cost / Ys

println("\n  Aggregate changes:")
@printf("    ΔK/K = %+.2f%%\n", 100*(Ks-Kb)/Kb)
@printf("    ΔH/H = %+.2f%%\n", 100*(Hs-Hb)/Hb)
@printf("    ΔY/Y = %+.2f%%\n", 100*(Ys-Yb)/Yb)
@printf("    Corr(k,z): %.4f → %.4f\n", res_base.corr, res_sub.corr)
@printf("    Subsidy cost / Y = %.2f%%\n", 100*cost_frac)

# ── Figure 4: Stationary distribution ────────────────────────
let
    kg   = res_base.k_grid
    dk   = kg[2] - kg[1]
    μk_b = vec(sum(res_base.μ; dims=2))
    μk_s = vec(sum(res_sub.μ;  dims=2))

    p4a = bar(kg, μk_b; bar_width=dk*0.85, color=COLORS[1], alpha=0.8,
              label="Baseline", xlabel="Capital k", ylabel="Density",
              title="Marginal distribution over k")
    plot!(p4a, kg, μk_s; color=COLORS[2], lw=2,
          seriestype=:steppost, label="Subsidy tau=0.10")

    # Weighted histogram via manual binning (no StatsBase needed)
    ir_b  = vec(inv_rate_mat(res_base.i_star, res_base.k_grid)) .* 100
    ir_s  = vec(inv_rate_mat(res_sub.i_star,  res_sub.k_grid))  .* 100
    wb    = vec(res_base.μ)
    ws    = vec(res_sub.μ)
    bins  = collect(-40.0:2.0:60.0)
    nb    = length(bins) - 1
    hist_b = zeros(nb); hist_s = zeros(nb)
    for i in 1:nb
        lo, hi    = bins[i], bins[i+1]
        hist_b[i] = sum(wb[(ir_b .>= lo) .& (ir_b .< hi)])
        hist_s[i] = sum(ws[(ir_s .>= lo) .& (ir_s .< hi)])
    end
    mids = (bins[1:end-1] .+ bins[2:end]) ./ 2

    p4b = bar(mids, hist_b; bar_width=1.8, color=COLORS[1], alpha=0.6,
              label="Baseline", xlabel="Investment rate i/k (%)",
              ylabel="Mass", title="Histogram of investment rates")
    bar!(p4b, mids, hist_s; bar_width=1.8, color=COLORS[2], alpha=0.5,
         label="Subsidy tau=0.10")
    vline!(p4b, [0]; color=:black, lw=0.8, label="")

    fig4 = plot(p4a, p4b; layout=(1,2), size=(1100,420),
                plot_title="Stationary Distribution μ(k, z)",
                left_margin=5mm, bottom_margin=5mm)
    savefig(fig4, "figures/fig4_stationary_dist.png")
    println("Saved figures/fig4_stationary_dist.png")
end

# ── Figure 5: Subsidy policy functions ───────────────────────
let
    kg     = res_base.k_grid
    nz_    = length(res_base.z_grid)
    iz_plt = [1, nz_÷2+1, nz_]
    ir_b   = inv_rate_mat(res_base.i_star, res_base.k_grid) .* 100
    ir_s   = inv_rate_mat(res_sub.i_star,  res_sub.k_grid)  .* 100

    p5a = plot(; xlabel="Capital k", ylabel="i*/k (%)",
                 title="Policy i*(k,z)\nsolid: baseline, dashed: subsidy",
                 titlefontsize=9, legend=:topright)
    p5b = plot(; xlabel="Capital k", ylabel="Δ(i*/k) (ppt)",
                 title="Change in investment rate\ndue to subsidy",
                 titlefontsize=9, legend=:topright)

    for (ci, iz) in enumerate(iz_plt)
        lbl = "z=$(round(res_base.z_grid[iz]; digits=2))"
        plot!(p5a, kg, ir_b[:,iz]; color=COLORS[ci], lw=2, ls=:solid, label=lbl)
        plot!(p5a, kg, ir_s[:,iz]; color=COLORS[ci], lw=2, ls=:dash,  label="")
        plot!(p5b, kg, ir_s[:,iz] .- ir_b[:,iz]; color=COLORS[ci], lw=2, label=lbl)
    end
    hline!(p5a, [0]; color=:black, lw=0.7, ls=:dot, label="")
    hline!(p5b, [0]; color=:black, lw=0.8, label="")

    fig5 = plot(p5a, p5b; layout=(1,2), size=(1100,420),
                plot_title="Policy Analysis: Investment Subsidy tau = 0.10",
                left_margin=5mm, bottom_margin=5mm)
    savefig(fig5, "figures/fig5_subsidy_policy.png")
    println("Saved figures/fig5_subsidy_policy.png")
end

# ── Figure 6: VFI convergence & grid sensitivity ─────────────
println("\n  Running convergence diagnostics …")

function solve_with_diffs(γ, F, ps; nk=45, nz=7, tol=1e-6, maxiter=500)
    p       = FirmModel.P
    zlog, Pi = FirmModel.tauchen(p.rho, p.sigma; n=nz)
    z_grid  = exp.(zlog) .* p.zbar
    k_grid  = FirmModel.make_k_grid(p, nk)
    β       = p.beta
    π̃ = [FirmModel.static_profit(z_grid[iz], k_grid[ik])
          for iz in 1:nz, ik in 1:nk]
    V     = [π̃[iz,ik]/(1-β) for ik in 1:nk, iz in 1:nz]
    EV    = zeros(nk, nz)
    V_new = zeros(nk, nz)
    diffs = Float64[]
    for _ in 1:maxiter
        V_old = copy(V)
        mul!(EV, V_old, Pi')
        fill!(V_new, -Inf)
        for ik in 1:nk
            k        = k_grid[ik]
            i_vec    = k_grid .- (1 - p.delta)*k
            pv       = FirmModel.price_invest(i_vec, Float64(ps), 0.0)
            cv       = FirmModel.adj_cost_vec(i_vec, k, Float64(γ), Float64(F))
            cost_vec = pv .* i_vec .+ cv
            for iz in 1:nz
                obj = π̃[iz,ik] .- cost_vec .+ β .* EV[:,iz]
                V_new[ik,iz] = maximum(obj)
            end
        end
        diff = maximum(abs.(V_new .- V_old))
        push!(diffs, diff)
        V .= V_new
        diff < tol && break
    end
    return diffs
end

diffs_conv   = solve_with_diffs(γ_hat, F_hat, ps_hat)
nk_grids     = [20, 30, 45]
avg_inv_grid = [run_pipeline(γ_hat, F_hat, ps_hat; nk=nk_g, nz=7).moments.avg_inv
                for nk_g in nk_grids]

let
    p6a = plot(1:length(diffs_conv), diffs_conv;
               yscale=:log10, color=COLORS[1], lw=2, legend=false,
               xlabel="VFI Iteration", ylabel="|ΔV| (log scale)",
               title="VFI Convergence (calibrated baseline)")

    p6b = plot(nk_grids, avg_inv_grid .* 100;
               color=COLORS[2], lw=2, marker=:circle, markersize=8,
               xlabel="Capital grid points nk", ylabel="Avg investment rate (%)",
               title="Grid sensitivity: avg i/k vs nk",
               xticks=nk_grids, legend=:topright)
    hline!(p6b, [DATA_MOMENTS[1]*100]; color=:grey, ls=:dash, lw=1.2,
           label="Data target")

    fig6 = plot(p6a, p6b; layout=(1,2), size=(1100,420),
                plot_title="Accuracy: VFI Convergence & Grid Sensitivity",
                left_margin=5mm, bottom_margin=5mm)
    savefig(fig6, "figures/fig6_accuracy.png")
    println("Saved figures/fig6_accuracy.png")
end


# Summary table


println("\n" * "═"^60)
println("  SUMMARY RESULTS")
println("═"^60)
@printf("\n  Estimated parameters:\n")
@printf("    γ̂  = %.4f  (convex cost magnitude)\n",  γ_hat)
@printf("    F̂  = %.5f  (fixed cost, fraction of k)\n", F_hat)
@printf("    p̂ₛ = %.4f  (resale price / irreversibility)\n", ps_hat)

println("\n  Moment fit (data vs model):")
for (l, d, m) in zip(mom_labels, DATA_MOMENTS, m_hat)
    @printf("    %s  %5.1f%%   %5.1f%%\n", l, 100d, 100m)
end

println("\n  Subsidy effects (tau = 0.10 vs tau = 0):")
@printf("    ΔK/K = %+.2f%%\n", 100*(Ks-Kb)/Kb)
@printf("    ΔH/H = %+.2f%%\n", 100*(Hs-Hb)/Hb)
@printf("    ΔY/Y = %+.2f%%\n", 100*(Ys-Yb)/Yb)
@printf("    Corr(k,z): %.4f → %.4f\n", res_base.corr, res_sub.corr)
@printf("    Subsidy cost / Y = %.2f%%\n", 100*cost_frac)

println("\n  Investment moments with subsidy:")
for (l, mb, ms) in zip(mom_labels, m_hat, m_sub)
    @printf("    %s  base=%5.1f%%  subsidy=%5.1f%%\n", l, 100mb, 100ms)
end

println("\nAll figures are in ./figures/")
println("═"^60)
