using Random, Distributions, StatsPlots, StatsBase, LinearAlgebra, DataFrames, Printf
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
