using Random, Statistics, Distributions, Plots, DataFrames, LaTeXStrings, CSV, LinearAlgebra, IterativeSolvers, PrettyTables

# Problem 2
function create_matrices(α, β)
    A = [1   -1   0   α-β   β;
         0    1  -1    0    0;
         0    0   1   -1    0;
         0    0   0    1   -1;
         0    0   0    0    1]
    b = [α; 0; 0; 0; 1]
    return A, b
end

function solve_sys(A, x, b, α, β)
    n = length(b)
    for i in n:-1:1
        x[i] = (b[i] - sum(A[i,j] * x[j] for j in (i+1):n; init=0.0)) / A[i,i]
    end
    return x
end

function relative_residual_norm(A, x, b)
    r = b - A*x
    return norm(r) / norm(b)
end

function solve_sys_and_bs(A, b, α, β)
    x_bs = A \ b
    x_solve_sys = solve_sys(A, zeros(5), b, α, β)
    r_r_bs = relative_residual_norm(A, x_bs, b)
    r_r_solve_sys = relative_residual_norm(A, x_solve_sys, b)
    cond_A = cond(A)
    return x_bs, x_solve_sys, r_r_bs, r_r_solve_sys, cond_A
end

function results_table(α, β_values)
    results = DataFrame(
        β = Float64[],
        x1_exact = Float64[],
        x1_backslash = Float64[],
        Condition_Number = Float64[],
        Relative_Residual_Exact = Float64[],
        Relative_Residual_Backslash = Float64[]
    )
    
    for β in β_values
        A, b = create_matrices(α, β)
        x_bs, x_exact, r_r_bs, r_r_exact, cond_A = solve_sys_and_bs(A, b, α, β)
        push!(results, (β, x_exact[1], x_bs[1], cond_A, r_r_exact, r_r_bs))
    end
    
    return results
end


α = 0.1
β_values = [10.0^i for i in 0:12]  
results = results_table(α, β_values)
println(results)
pretty_table(results)

# The way that julia solves linear system using backslash operator and backward substituion (for this question) are equivalent since the given matrix is already in upper triangular form already.
# Condition number grows as κ(A) ≈ √2× β² after β = 10^11 we lose accuracy in the solution due to ill-conditioning of the matrix. while expected result is 1 we get 0.999994 for β = 10^11 and 1.00002 for β = 10^12