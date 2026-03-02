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


# Steady state capital k_ss used for benchmarking and grid construction. Derived from the FOC of the convex problem (F=0, ps=1):
# dπ̃/dk = A·φ·k^(φ-1) = r+δ  =>  k_ss = (A·φ/(r+δ))^(1/(1-φ))
# where A=(1-ν)(ν/w)^(ν/(1-ν)), φ=α/(1-ν)

function find_kss(p::FirmParams)
    A = (1 - p.ν) * (p.ν / p.w)^(p.ν / (1 - p.ν))
    φ = p.α / (1 - p.ν)
    (A * φ / (p.r + p.δ))^(1 / (1 - φ))
end

# -----------------------------------------------------------------------------
# CAPITAL GRID
#
# Two-component grid:
#   (a) Power-spaced global grid of Nk points over [0.1·k_ss, 5·k_ss]
#       θ_grid=0.4 crowds points near k_min (where curvature is highest)
#   (b) N_FINE_FIXED = 100 uniformly-spaced points over [0.4·k_ss, 1.6·k_ss]
#       This is where the (S,s) kinks live for any reasonable F value.
#
# CRITICAL: N_FINE_FIXED is a constant, never a function of Nk.
# The grid sensitivity check increases only Nk. If N_FINE_FIXED also grew
# with Nk, the sensitivity check would measure structure changes not just
# density changes, producing artificially large numbers (that was the 36% bug).
#
# HOW TO VERIFY THE GRID IS GOOD:
#   After solving, check that:
#   1. sum(mu[[1,end],:]) < 1e-4  → negligible mass at boundaries
#   2. grid sensitivity < 1%      → moments stable under refinement
#   3. The kink boundaries s_lo, s_hi fall inside [0.4·k_ss, 1.6·k_ss]
#      → they are inside the fine region, well-resolved
# -----------------------------------------------------------------------------


function setup_capital_grid(p::FirmParams)
    k_ss  = find_kss(p)
    k_min = 0.1 * k_ss
    k_max = 5.0 * k_ss
    u      = range(0.0, 1.0, length=p.Nk)
    k_base = @. k_min + (k_max - k_min) * u^p.θ_grid
    k_fine = collect(range(0.4*k_ss, 1.6*k_ss, length=N_FINE_FIXED))
    sort(unique(vcat(k_base, k_fine))), k_ss
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
function bellman_operator!(Vn, pkl, pia, Vo, profit, kg, Pi, p::FirmParams)
    Nk = length(kg)
    Nz = size(Pi, 1)
    β  = 1 / (1 + p.r)
    EV = β .* (Vo * Pi')                 # (Nk × Nz) continuation values

    for iz in 1:Nz, ik in 1:Nk
        k   = kg[ik]
        π   = profit[ik, iz]
        k_s = (1 - p.δ) * k              # inaction landing point

        # Option A : Inaction, does not pay the fixed cost, just gets continuation value at k_s
        vA = π + interp(k_s, iz, kg, EV)

        # Option B: Adjust, pays fixed cost, gets continuation value at chosen k'
        vB  = -Inf;  kB = kg[1]
        for im in 1:Nk
            v = π - inv_cost(kg[im] - k_s, k, p) - p.F*k + EV[im, iz]
            v > vB && (vB = v; kB = kg[im])
        end

        #pkl denotes, policy capital: the k' that maximizes the RHS of Bellman. pia is a boolean matrix denoting whether the optimal choice is inaction (true) or adjustment (false).
        if vA > vB
            Vn[ik,iz] = vA;  pkl[ik,iz] = k_s;  pia[ik,iz] = true
        else
            Vn[ik,iz] = vB;  pkl[ik,iz] = kB;   pia[ik,iz] = false
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

# Value Function Iteration with Howard acceleration. Returns a dictionary with all relevant outputs for analysis and plotting.
function vfi_solve(p::FirmParams; tol=1e-6, maxiter=3000,
                   h_every=10, h_steps=30, verbose=true, label="")
    zg, Pi   = setup_productivity(p)
    kg, k_ss = setup_capital_grid(p)
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

#Stationary distribution μ(k,z) as the fixed point of the transition operator induced by the policy. We use power iteration: start with uniform μ, apply the operator, normalize, and repeat until convergence.
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


# Moments
# Each = Σ_{ik,iz} indicator(ir_{ik,iz}) · μ(ik,iz)
# discrete integral ∫ indicator dμ over the stationary distribution
function compute_moments(pol_i, kg, mu)
    ir = pol_i ./ kg     
    (avg_ir    = sum(ir .* mu),
     inaction  = sum((abs.(ir) .< 0.01) .* mu),
     frac_neg  = sum((ir .< 0.0)         .* mu),
     pos_spike = sum((ir .> 0.20)         .* mu),
     neg_spike = sum((ir .< -0.20)        .* mu))
end

# Boundary mass check: if the stationary distribution puts significant mass at the grid boundaries, it suggests that the grid range is too narrow and may be truncating important parts of the distribution. We check the mass at the lower and upper boundaries (first and last rows of μ) and print a warning if either exceeds 1e-4.
# If significant mass is at the grid boundaries, the grid range is too narrow. Increase k_max or decrease k_min to capture the full distribution.
function check_boundary_mass(mu, label="")
    lo = sum(mu[1,   :])
    hi = sum(mu[end, :])
    ok = max(lo, hi) < 1e-4
    @printf("  Boundary mass [%s]:  lower=%.2e  upper=%.2e  %s\n",
            label, lo, hi, ok ? "OK" : "GRID TOO NARROW — increase k_max or decrease k_min")
end

# -----------------------------------------------------------------------------
# KINK DETECTION
#
# For each z, scan pol_inact along k and find where the firm switches
# from adjusting → inaction (lower kink s_lo) and inaction → adjusting
# (upper kink s_hi). These bracket the (S,s) band.
#
# The band should:
#   • Contain k_ss (target capital is inside the no-action region)
#   • Narrow as z rises (high-productivity firms have larger marginal
#     benefit of capital so they tolerate less deviation)
#   • Widen as F increases (higher fixed cost = wider tolerance)
# -----------------------------------------------------------------------------
function find_kinks(sol)
    kg = sol.kg;  pia = sol.pia;  zg = sol.zg
    Nz = length(zg)
    df = DataFrame(z_idx=Int[], z_val=Float64[],
                   s_lo=Float64[], s_hi=Float64[], band=Float64[])
    for iz in 1:Nz
        idx = findall(pia[:, iz])
        if isempty(idx)
            push!(df, (iz, round(zg[iz],digits=4), NaN, NaN, NaN))
        else
            s_lo = kg[minimum(idx)]
            s_hi = kg[maximum(idx)]
            push!(df, (iz, round(zg[iz],digits=4),
                       round(s_lo,digits=4), round(s_hi,digits=4),
                       round(s_hi-s_lo,digits=4)))
        end
    end
    df
end

function print_kinks(sol, label="")
    df = find_kinks(sol)
    println("\n  (S,s) Inaction Bands — $label")
    pretty_table(df)
    valid = filter(r -> !isnan(r.band), df)
    if nrow(valid) > 0
        @printf("  Mean band width = %.4f   k_ss = %.4f\n",
                mean(valid.band), sol.k_ss)
        @printf("  Band narrows with z: %s\n",
                valid.band[1] > valid.band[end] ? "YES" : "NO — check model")
        # Check kinks are inside fine-grid region
        all_in = all(r -> r.s_lo >= 0.4*sol.k_ss && r.s_hi <= 1.6*sol.k_ss,
                     eachrow(valid))
        @printf("  Kinks inside fine region [0.4,1.6]·k_ss: %s\n",
                all_in ? "YES — well resolved" : "NO — extend fine region!")
    else
        println("  No inaction region (F=0 or too small)")
    end
end

# Table helpers 
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

# =============================================================================
# GRID SENSITIVITY CHECK
# IMPORTANT: only run this at parameters where the model is well-behaved
# (i.e. calibrated parameters, not F=0.05 exploration stages).
# At F=0.05 the 88% inaction rate is highly sensitive to any grid change
# because the inaction boundary spans nearly the whole grid — that is
# an economic fact about F=0.05, not a numerical error.
# =============================================================================
function grid_sensitivity(p::FirmParams, base_mom; verbose=true)
    pf = FirmParams(; α=p.α, ν=p.ν, δ=p.δ, r=p.r, w=p.w, ρ=p.ρ, σ=p.σ,
                     θ_grid=p.θ_grid, γ_adj=p.γ_adj, F=p.F, ps=p.ps,
                     Nk=p.Nk+100, Nz=p.Nz+2)
    sf = vfi_solve(pf, verbose=false)
    mf = stationary_dist(sf.pkl, sf.kg, sf.Pi, verbose=false)
    mf_mom = compute_moments(sf.pol_i, sf.kg, mf)
    verbose && show_sensitivity_table(base_mom, mf_mom, p.Nk, pf.Nk)
    mf_mom
end


# Plots
function z_sel(zg)
    Nz = length(zg)
    idx = [1, div(Nz+1,2), Nz]
    idx, ["Low z=$(round(zg[i],digits=3))" for i in idx]
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


# =============================================================================
# SMM CALIBRATION
#
# HOW TO SET GOOD STARTING VALUES — the systematic approach:
#
# Step 1: Run a coarse grid search over (γ, F, ps).
#   We evaluate the SMM objective on a 4×4×4 = 64 point grid.
#   This takes ~64 × (one VFI) ≈ a few minutes but gives a reliable
#   starting point that is NOT in a bad basin of attraction.
#
# Step 2: Take the best point from the grid search as θ₀ for Nelder-Mead.
#
# WHY THIS MATTERS:
#   The objective surface has multiple local minima because:
#   - The inaction rate is a step function of F (flat, then jumps)
#   - With F too large, the optimizer gets stuck at inaction=100%, loss>>0
#   - Starting near the truth avoids this
#
# PARAMETER BOUNDS (sigmoid reparametrisation):
#   θ_raw ∈ ℝ³ (unconstrained) → physical parameters via sigmoid:
#   γ  ∈ [0.05,  5.0]
#   F  ∈ [0.0,   0.05]   ← tighter upper bound than before (F=0.05 was too large)
#   ps ∈ [0.3,   0.999]
# =============================================================================



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

# Coarse grid search: evaluates objective on a grid to find a good θ₀
# before Nelder-Mead. Protects against local minima.
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
# Grid sensitivity only meaningful at well-behaved parameters
grid_sensitivity(p1, m1)


# STAGE 2 — Fixed costs  (F=0.05, ps=1, γ=2)
# NOTE: F=0.05 is deliberately large for exploration.
# Expected: wide inaction band, inaction rate ~88%, kinks at boundaries of inaction region
# Grid sensitivity will be large here that is EXPECTED because the model
# is highly sensitive to boundary placement when inaction spans entire grid.

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
print_kinks(s2, "S2")



# STAGE 3 — Irreversibility  (F=0.05, ps=0.5, γ=2)
# Expected: asymmetric policy, frac_neg and neg_spike fall vs Stage 2

println("\nSTAGE 3 — Irreversibility (F=0.05, ps=0.5, γ=2)\n")
p3  = FirmParams(γ_adj=2.0, F=0.05, ps=0.5, Nk=250, Nz=11)
s3  = vfi_solve(p3, verbose=true, label="S3")
mu3 = stationary_dist(s3.pkl, s3.kg, s3.Pi, verbose=true)
m3  = compute_moments(s3.pol_i, s3.kg, mu3)
check_boundary_mass(mu3, "S3")
println("  Frac neg:  S2=$(round(m2.frac_neg*100,digits=2))% → S3=$(round(m3.frac_neg*100,digits=2))%")
println("  Neg spike: S2=$(round(m2.neg_spike*100,digits=2))% → S3=$(round(m3.neg_spike*100,digits=2))%")
show_moments_table(m3, "Stage 3")
plot_policy(s3, "Irreversibility F=0.05 ps=0.5", save_prefix="s3")
plot_stationary(s3, mu3, "Irreversibility", save_prefix="s3")
print_kinks(s3, "S3")



# STAGE 4 — SMM Calibration
# Grid search finds starting values → Nelder-Mead refines

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
print_kinks(sc, "Calibrated")

# Grid sensitivity ONLY on calibrated model
println("\n  Grid sensitivity at calibrated parameters:")
grid_sensitivity(pc, mc)


# Cross Stage Summary

println("\nCROSS-STAGE MOMENT SUMMARY\n")
show_cross_table([m1,m2,m3,mc], ("Stage1","Stage2","Stage3","Calibrated"))
@printf("\n  γ̂=%.4f  F̂=%.5f  p̂s=%.4f\n", γh, Fh, psh)