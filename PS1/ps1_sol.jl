using Random, Distributions, StatsPlots, StatsBase, LinearAlgebra, DataFrames, Printf, Roots, LaTeXStrings, Plots

pyplot()  # Swithced backend due to dodgy rendering of LaTeX labels in default backend.

Random.seed!(0)

# ============== Problem 1 ==================
function get_pois_z(n, nrep=1000, λ=1)
    X = rand(Poisson(λ), n, nrep)     # n × nrep
    x̄ = vec(mean(X, dims = 1))
    μ = λ
    σ = sqrt(λ)
    z = (x̄ .- μ) .* sqrt(n) ./ σ
    return z
end

function plot_pois(z, n)
    p = histogram(
        z;
        bins = -4:0.5:4,
        normalize = :pdf,
        legend = false,
        xlabel = "z",
        ylabel = "Frequency",
        title = "N = $n",
        alpha = 1,
    )
    xlims!(-4, 4)
    ylims!(0, 1)
    z_range = -4:0.01:4
    plot!(z_range, pdf.(Normal(0, 1), z_range), lw = 2, color=:red)
    
    return p
end

λ = 1.0
μ = λ
σ = sqrt(λ)


ns = [5, 25, 100, 1000]
nrep = 1000
z_range = -4:0.01:4

plots = Any[]

for n in ns
    z = get_pois_z(n, nrep, λ)

    p = plot_pois(z, n)
    push!(plots, p)
end

plot(plots..., layout = (2, 2))

# savefig("PS1/figure/ps1_problem1.png")

# ============== Problem 2 ==================

# Solution by hand is provided in "PS1/PS 1 - Task 2.pdf"
function linsolve_exact(α, β)
    return [1.0; 1.0; 1.0; 1.0; 1.0]
end

function linsolve_backslash(α, β)
    A = [
        1.0 -1.0 0.0 α-β β;
        0.0 1.0 -1.0 0.0 0.0;
        0.0 0.0 1.0 -1.0 0.0;
        0.0 0.0 0.0 1.0 -1.0;
        0.0 0.0 0.0 0.0 1.0
    ]
    b = [α;0.0;0.0;0.0;1.0]

    return A \ b
end

function relative_residual(x_exact, x_approx)
    return abs.(x_exact - x_approx) ./ abs.(x_exact)
end

alpha = 0.1
test_betas = [10.0^i for i in 0:50]  # β ≥ 10^19 will lead to integer overflow. 
                                     # Using floating point numbers instead.
formatted_betas = [@sprintf("10^%d", i) for i in 0:50]

sol_exact = [linsolve_exact(alpha, β) for β in test_betas]
sol_backslash = [linsolve_backslash(alpha, β) for β in test_betas]

x1_exact = [sol[1] for sol in sol_exact]
x1_backslash = [sol[1] for sol in sol_backslash]

relative_resid = [relative_residual(sol_exact[i], sol_backslash[i]) for i in 1:length(test_betas)]

cond_num = [cond([1.0 -1.0 0.0 alpha-β β;
                   0.0 1.0 -1.0 0.0 0.0;
                   0.0 0.0 1.0 -1.0 0.0;
                   0.0 0.0 0.0 1.0 -1.0;
                   0.0 0.0 0.0 0.0 1.0]) for β in test_betas]


df = DataFrame(Beta = formatted_betas,
                x1_exact = x1_exact,
                x1_backslash = x1_backslash,
                rel_resid_x1 = [rel[1] for rel in relative_resid],
                cond_num = cond_num)

# On the pattern of condition number relative to β:
#
#   For higher β (≈10^4 and above), a clear pattern emerges in cond(A) relative to β.
#   We see that cond(A) ≈ √2 * β^2 = √2 * 10^(2i) for i ∈ [4, ∞). 
#   
#   In this case, the "\" operator is able hold perfect accuracy 
#   for the entire range tested (i ∈ [0, 50])
#
#   In this example, A and b are both well-conditioned and very simple in structure.
#   Therefore, the condition number (which measures a "worst-case" sensitivity) becomes
#   less relevant for predicting numerical accuracy of the solution.
#
#   Despite an exponentially increasing cond(A), the "\" operator maintains exact accuracy
#   over the entire range of β tested.

# ============== Problem 3 ==================
function p1_objective(p_1, α_1, σ_1=0.2, σ_2=0.2)
    if α_1 <= 0.0 || α_1 >= 1.0
        error("α_1 must be in (0,1)")
    end
    
    α_2 = 1.0 - α_1

    p_2 = 1.0  # Normalize p_2 and solve for p_1

    w_1 = [1.0, 1.0]
    w_2 = [0.5, 1.5]

    alpha = [α_1, α_2]
    sigma = [σ_1, σ_2]
    w = [w_1, w_2]

    lhs = 0.0
    rhs = 0.0

    for i in 1:2
        lhs_num = (alpha[i]^sigma[i]*p_1^(-sigma[i])) 
        lhs_denom = (alpha[i]^sigma[i]*p_1^(1-sigma[i]) + (1 - alpha[i])^sigma[i]*p_2^(1-sigma[i])) 
        m_i = (p_1 * w[i][1] + p_2 * w[i][2])

        lhs += (lhs_num / lhs_denom) * m_i
        

        rhs += w[i][1]
    end
    return lhs - rhs
end

# Find root by bisection (f(a) * f(b) ≤ 0)
function bisect(f, a, b; tol=1e-12, max_iter=1000)
    fa = f(a)
    fb = f(b)

    if fa * fb > 0
        @warn "f(a) and f(b) must have opposite signs. f(a) = $fa, f(b) = $fb"
    end

    for i in 1:max_iter
        # Midpoint
        c = (a + b) / 2
        fc = f(c)

        # check for early stopping
        if abs(fc) < tol || (b - a) / 2 < tol
            return c
        end

        # preserve sign change
        if fa * fc < 0
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
    end
    error("Maximum iterations reached without convergence")
end

function q1(p1, α1)
    α2 = 1.0 - α1
    p2 = 1.0

    w1 = [1.0, 1.0]
    w2 = [0.5, 1.5]

    q1 = 0.0
    for i in 1:2
        alpha = [α1, α2]
        w = [w1, w2]

        m_i = (p1 * w[i][1] + p2 * w[i][2])
        q1 += (alpha[i]^0.2 * p1^(-0.2) / (alpha[i]^0.2 * p1^(0.8) + (1 - alpha[i])^0.2 * p2^(0.8))) * m_i
    end
    return q1
end

function consumption(p, α, σ1)
    p2 = 1.0
    w1 = [1.0, 1.0]
    w2 = [0.5, 1.5]
    α2 = 1.0 - α
    σ2 = σ1

    lhs_num_c1 = (α^σ1*p^(-σ1))
    lhs_denom_c1 = (α^σ1*p^(1-σ1) + (1 - α)^σ1*p2^(1-σ1))
    m_i1 = (p * w1[1] + p2 * w1[2])

    c1_1 = (lhs_num_c1 / lhs_denom_c1) * m_i1
    
    lhs_num_c2 = (α2^σ2*p^(-σ2))
    lhs_denom_c2 = (α2^σ2*p^(1-σ2) + (1 - α2)^σ2*p2^(1-σ2))
    m_i2 = (p * w2[1] + p2 * w2[2])

    c2_1 = (lhs_num_c2 / lhs_denom_c2) * m_i2



    return c1_1, c2_1
end


α1_values = 0.01:0.0001:0.99

# σ = 0.2 case
p1_solutions_02= Float64[]
q1_values_02 = Float64[]
c1_1_values_02 = Float64[]
c2_1_values_02 = Float64[]
for α1 in α1_values
    obj = p -> p1_objective(p, α1)
    p1_sol = bisect(obj, 0.01, 1000.0; tol=1e-12, max_iter=1000)
    push!(p1_solutions_02, p1_sol)

    q1_val = q1(p1_sol, α1)
    push!(q1_values_02, q1_val)

    c1_1_val, c2_1_val = consumption(p1_sol, α1, 0.2)
    push!(c1_1_values_02, c1_1_val)
    push!(c2_1_values_02, c2_1_val)
end

df_02 = DataFrame(α1 = α1_values,
                p1 = p1_solutions_02,
                q1 = q1_values_02,
                c1_1 = c1_1_values_02,
                c2_1 = c2_1_values_02)

# sigma = 5.0 case
p1_solutions_5 = Float64[]
q1_values_5 = Float64[]
c1_1_values_5 = Float64[]
c2_1_values_5 = Float64[]
for α1 in α1_values
    obj = p -> p1_objective(p, α1, 5.0, 5.0)
    p1_sol = bisect(obj, 0.01, 1000.0; tol=1e-12, max_iter=1000)
    push!(p1_solutions_5, p1_sol)

    q1_val = q1(p1_sol, α1)
    push!(q1_values_5, q1_val)

    c1_1_val, c2_1_val = consumption(p1_sol, α1, 5.0)
    push!(c1_1_values_5, c1_1_val)
    push!(c2_1_values_5, c2_1_val)
end

df_5 = DataFrame(α1 = α1_values,
                  p1 = p1_solutions_5,
                  q1 = q1_values_5,
                  c1_1 = c1_1_values_5,
                  c2_1 = c2_1_values_5)

plots_p = Any[]

# Price plots
p1_02 = plot(
    df_02.α1,
    df_02.p1;
    xlabel = L"\alpha_1",
    ylabel = L"p_1",
    title = "σ = 0.2",
    titlefontsize=11,
    legend = false,
)

p1_5 = plot(
    df_5.α1,
    df_5.p1;
    xlabel = L"\alpha_1",
    ylabel = L"p_1",
    title = "σ = 5.0",
    titlefontsize=11,
    legend = false,
)

push!(plots_p, p1_02)
push!(plots_p, p1_5)
plot(plots_p..., 
    layout = (1, 2), 
    size=(800, 400), 
    plot_title="Equilibrium " * L"p_1" * " vs. " * L"\alpha_1", 
    plot_titlefontsize=14,
    )

savefig("PS1/figure/ps1_problem3_price.png")

# Consumption Plots
plots_c = Any[]

c1 = plot(
    df_02.α1,
    df_02.c1_1;
    xlabel = L"\alpha_1",
    ylabel = L"c_{i,1}",
    label = L"c_{1,1}",
    title = "σ = 0.2",
    titlefontsize=11,
    legend = false,
)
plot!(df_02.α1, df_02.c2_1; lw=2, label=L"c_{2,1}")
push!(plots_c, c1)

c2 = plot(
    df_5.α1,
    df_5.c1_1;
    xlabel = L"\alpha_1",
    ylabel = L"c_{i,1}",
    label = L"c_{1,1}",
    title = "σ = 5.0",
    titlefontsize=11,
    legend = :outertopright,
)
plot!(df_5.α1, df_5.c2_1; lw=2, label=L"c_{2,1}")
push!(plots_c, c2)

plot(plots_c..., 
    layout = (1, 2), 
    size=(800, 400), 
    plot_title="Equilibrium " * L"c_{i,1}" * " vs. " * L"\alpha_1", 
    plot_titlefontsize=14,
    )
savefig("PS1/figure/ps1_problem3_consumption.png")

df_02[!, :diff] = df_02.c1_1 .- df_02.c2_1
idx_02 = argmin(abs.(df_02.diff))
eq_02 = df_02[idx_02, [:α1, :c1_1, :c2_1, :diff]]

df_5[!, :diff] = df_5.c1_1 .- df_5.c2_1
idx_5 = argmin(abs.(df_5.diff))
eq_5 = df_5[idx_5, [:α1, :c1_1, :c2_1, :diff]]

println("Approximate equilibria within α grid increments of 0.0001:")
println("For σ = 0.2:\n")

print(eq_02, "\n\n")
println("For σ = 5.0:\n")
print(eq_5)


# ============== Problem 4 ==================
