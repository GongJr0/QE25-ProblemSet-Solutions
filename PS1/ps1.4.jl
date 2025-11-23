using Random, Statistics, Distributions,Plots,DataFrames,LaTeXStrings,CSV,Distributions,LinearAlgebra,IterativeSolvers,PrettyTables
pyplot()

#Problem 4 

function initialize_matrix(μ,Σ,n,target_return)
    A = zeros(Float64, n+2, n+2)
    b = zeros(n+2)
    b[n+1] = target_return
    b[n+2] = 1.0
    A[1:n, 1:n] = Σ
    A[1:n, n+1] = μ
    A[1:n, n+2] .= 1.0
    A[n+1, 1:n] = μ'
    A[n+2, 1:n] .= 1.0
    return A,b
end

function relative_residual_norm(A, x, b)
    r = b-A*x
    return norm(r) / norm(b)
end

function solve_portfolio_backslash(A, b, n)
    time = @elapsed x = A \ b
    portfolio_weights = x[1:n]
    lagrange_return = x[n+1]
    lagrange_weight = x[n+2]
    iterations = 1  
    return portfolio_weights, lagrange_return, lagrange_weight, iterations, time,x
end


function solve_portfolio_cg(A, b, n; x0=zeros(n+2))
    if issymmetric(A) && isposdef(A)
        x = copy(x0)
        time = @elapsed history = IterativeSolvers.cg!(x, A, b, log=true)
        iterations = history[2].iters
        
        portfolio_weights = x[1:n]
        lagrange_return = x[n+1]
        lagrange_weight = x[n+2]
        return portfolio_weights, lagrange_return, lagrange_weight, iterations, time, x
    else
        println("CG method couldn't be applied")
        return nothing, nothing, nothing, 0, 0.0, nothing
    end
end

function solve_portfolio_gmres(A, b, n; x0=zeros(n+2))
    x = copy(x0)
    time = @elapsed history = IterativeSolvers.gmres!(x, A, b, log=true)
    iterations = history[2].iters
    
    portfolio_weights = x[1:n]
    lagrange_return = x[n+1]
    lagrange_weight = x[n+2]
    return portfolio_weights, lagrange_return, lagrange_weight, iterations, time, x
end


#4.1 and 4.2
cd(@__DIR__)
markowitz = CSV.read("asset_returns.csv", DataFrame)
Σ = cov(Matrix(markowitz))
n = 500
μ = mean.(eachcol(markowitz))
target_return = 0.10

#4.3
A,b = initialize_matrix(μ,Σ,n,target_return)
condition_number = cond(A, 2)

#4.4

#4.4.a
portfolio_weights_bs, lagrange_return, lagrange_weight, iter_bs, time_bs, x_backslash = solve_portfolio_backslash(A, b, n)

#4.4.b
A_normal = A'*A
b_normal = A'*b
conditiion_number_normal = cond(A_normal, 2)
zeros_at_diagonal = any(diag(A_normal) .== 0.0)
is_diag_dominant = all(abs(A_normal[i,i]) > sum(abs.(A_normal[i,j]) for j in 1:size(A_normal,1) if j != i) for i in 1:size(A_normal,1))
#Matrix does not have zeros at diagonal but its not diagonally dominant thus we cant implement gauss-seidel method

#4.4.c
portfolio_weights_cg, lagrange_return_cg, lagrange_weight_cg, iter_cg, time_cg, x_cg = solve_portfolio_cg(A_normal, b_normal, n)

#4.4.d
portfolio_weights_gmres, lagrange_return_gmres, lagrange_weight_gmres, iter_gmres, time_gmres,x_gmres = solve_portfolio_gmres(A,b,n)

# 4.4.e
    P = zeros(Float64, n+2, n+2)
    P[1:n, 1:n] = Diagonal(Σ)
    P[n+1, n+1] = 1.0
    P[n+2, n+2] = 1.0
    P_inv = inv(P)

# Manually apply preconditioning
A_precond = P_inv * A
b_precond = P_inv * b
    

portfolio_weights_precond, lagrange_return_precond, lagrange_weight_precond, iter_prec, time_prec, x_precond = solve_portfolio_gmres(A_precond, b_precond, n)

backslah_n_err = relative_residual_norm(A, x_backslash, b)
cg_n_err = relative_residual_norm(A_normal, x_cg, b_normal)
gmres_n_err = relative_residual_norm(A, x_gmres, b)
precond_n_err = relative_residual_norm(A, x_precond, b)


#5
results = DataFrame(
    Method = ["Backslash (\\)", "CG (A'A)", "GMRES", "GMRES + Precond"],
    Iterations = [iter_bs, iter_cg, iter_gmres, iter_prec],
    Time_seconds = [time_bs, time_cg, time_gmres, time_prec],
    relative_residual_norm = [backslah_n_err, cg_n_err, gmres_n_err, precond_n_err]
)

println(results)
pretty_table(results)

#6
pws = [portfolio_weights_bs, portfolio_weights_cg, portfolio_weights_gmres, portfolio_weights_precond]
method_names = ["Backslash", "CG", "GMRES", "GMRES+Precond"]


weight_sums = Float64[]
expected_returns = Float64[]
variances = Float64[]
std_devs = Float64[]
weight_ok = String[]
return_ok = String[]

for (i, p) in enumerate(pws)
    wsum = sum(p)
    eret = dot(p, μ)
    var = p' * Σ * p
    stdev = sqrt(var)

    push!(weight_sums, wsum)
    push!(expected_returns, eret)
    push!(variances, var[])
    push!(std_devs, stdev[])
    push!(weight_ok, isapprox(wsum, 1.0, atol=1e-8) ? "✓" : "✗")
    push!(return_ok, isapprox(eret, target_return, atol=1e-8) ? "✓" : "✗")
end

markowitz_results = DataFrame(
    Method = method_names,
    Sum_of_Weights = weight_sums,
    Expected_Return = expected_returns,
    Variance = variances,
    Std_Dev = std_devs,
    Weights_OK = weight_ok,
    Return_OK = return_ok
)

pretty_table(markowitz_results)
# For the CG method variance and expected return are also slightly off comparing to the other methods this is likely due to the ill-conditioning of the normal equations matrix A'A. Condition number becomes cond(A)^2 thus constraint on expected return is not satisfied as accurately as the other methods.

#6.1
target_returns = Float64[]
std_devs = Float64[]
iterations_list = Int[]
actual_returns = Float64[]

x_prev = randn(n+2)*0.01 

ert_range = range(0.01, 0.10, length=50)  

for (idx, target_μ) in enumerate(ert_range)
    A, b = initialize_matrix(μ, Σ, n, target_μ)
    
    portfolio_weights, _, _, iterations, time, x_prev = solve_portfolio_gmres(A, b, n; x0=x_prev)
    
    var = portfolio_weights' * Σ * portfolio_weights
    stdev = sqrt(var[])
    actual_return = dot(portfolio_weights, μ)
    
    push!(target_returns, target_μ)
    push!(std_devs, stdev[])
    push!(iterations_list, iterations)
    push!(actual_returns, actual_return)

end
efficient_frontier = DataFrame(
    Target_Return = target_returns,
    Std_Dev = std_devs,
    Iterations = iterations_list,
    Actual_Return = actual_returns
)
pretty_table(efficient_frontier)

scatter(efficient_frontier.Std_Dev, efficient_frontier.Target_Return,
        xlabel=L"\sigma_p",
        ylabel=L"\bar{\mu_p}",
        title="Efficient Frontier",
        markerstrokewidth=0.5,
        legend=false)
savefig("figures/efficient_frontier_scatter.png")