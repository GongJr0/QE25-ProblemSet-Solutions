using Random, Distributions, StatsPlots, StatsBase, LinearAlgebra, DataFrames, Printf, Roots
Random.seed!(0)

# ============== Problem 1 ==================
function get_pois_z(n, nrep=1000, λ=1)
    X = rand(Poisson(λ), n, nrep)     # n×nrep
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
function get_p1(p_1, σ_1=0.2, σ_2=0.2)

    α_2 = 0.5  # initial guess
    α_1 = 1.0 - α_2

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

        lhs += (lhs_num / lhs_denom * m_i)/p_1
        

        rhs += w[i][1]
    end
    return lhs - rhs
end

# Find root by bisection (f(a) * f(b) ≤ 0)
function bisect(f, a, b; tol=1e-12, max_iter=1000)
    fa = f(a)
    fb = f(b)

    if fa * fb > 0
        error("f(a) and f(b) must have opposite signs")
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


p_1_sol_exact = find_zero(p -> get_p1(p), 0.1)

p_1_sol_bisect = bisect(p -> get_p1(p), 0.1, 100.0)

relative_error = abs(p_1_sol_exact - p_1_sol_bisect) / abs(p_1_sol_exact)

println("Exact solution for p_1: ", p_1_sol_exact)
println("Bisection solution for p_1: ", p_1_sol_bisect)
println("Relative error between exact and bisection solutions: ", relative_error)