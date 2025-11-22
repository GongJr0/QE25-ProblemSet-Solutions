using Statistics, DataFrames, CSV, IterativeSolvers, LinearAlgebra, Plots, Suppressor
pyplot()  # Swithced backend due to dodgy rendering of LaTeX labels in default backend.

# ============== Problem 4 ==================

# load DataFrame
asset_returns = DataFrame(CSV.File(joinpath(@__DIR__, "asset_returns.csv")))

function get_system(μ̄, returns=asset_returns, init_weights=nothing)

    μ = []
    for col in names(returns)
        push!(μ, mean(returns[!, col]))
    end
    Σ = cov(Matrix(returns))
    n = size(Σ, 1)

    opt_matrix = zeros(n + 2, n + 2)
    for i in 1:n
        for j in 1:n
            opt_matrix[i, j] = Σ[i, j]
        end
        opt_matrix[i, end-1] = μ[i]
        opt_matrix[i, end] = 1.0
        opt_matrix[end-1, i] = μ[i]
        opt_matrix[end, i] = 1.0
    end

    if init_weights === nothing
        weight_vec = fill(1.0 / n, n) # equal weights if no guess
    else
        weight_vec = init_weights
    end
    
    λ_1 = 0.0
    λ_2 = 0.0  # λ_2 = 1 because the current sum of weights is already 1.0

    unknown_vec = vcat(weight_vec, λ_1, λ_2)
    rhs_vec = vcat(zeros(n), μ̄, 1.0)
    return opt_matrix, rhs_vec, n
end

function solve_backslash(μ̄, returns=asset_returns; return_lambda=false, init_weights=nothing)
    A, b, n = get_system(μ̄, returns, init_weights)
    x = A \ b

    if sum(x[1:n]) ≈ 1.0 == false
        @info "Weights sum to 1.0"
    end

    if !return_lambda
        return x[1:n]
    end
    return x
end

A, b, n = get_system(0.1, asset_returns)

AᵀA = A' * A

println("cond(A) = ", cond(A))  # ≈ 21626.13
println("cond(AᵀA) = ", cond(AᵀA))  # ≈ 4.6769e+8

# diag zeroes check
all(diag(AᵀA) .!= 0.0)  # true

# diagonal dominance check |a_i=j| = ∑|a_i≠j|
function diag_dominance(mat)
    n = size(mat, 1)
    row_check = Bool[]
    for i in 1:n
        diag = abs(mat[i,i])
        non_diag_sum = sum(abs.(mat[i, :])) - diag
        push!(row_check, diag >= non_diag_sum)
    end
    return all(row_check)
end

diag_dominance(AᵀA)  # false

# Diagonal dominance is not satisfied.
# Cannot implement Gauss-Seidel or Jacobi methods.

# Check sym and posdef
issymmetric(AᵀA) & isposdef(AᵀA)  # true

# Therefore, we can use Conjugate Gradient method.
function solve_cg(μ̄, returns=asset_returns; return_lambda=false, init_weights=nothing, verbose=false)
    A, b, n = get_system(μ̄, returns, init_weights)
    AᵀA = A' * A
    Aᵀb = A' * b
    x, ch = cg(AᵀA, Aᵀb, log=true)
    if ch.isconverged && verbose
        println("CG converged in ", ch.iters, " iterations.")
    elseif !ch.isconverged
        println("CG did not converge.") # Print non-convergence message regardless of verbosity
    end
    if sum(x[1:n]) ≈ 1.0 == false
        @info "Weights sum to 1.0"
    end

    if !return_lambda
        return x[1:n]
    end
    return x
end

function precond_gmres(A)
    Σ = A[1:end-2, 1:end-2]  # = Σ
    n = size(Σ, 1)

    P = zeros(n + 2, n + 2)
    for i in 1:n
        P[i, i] = Σ[i, i]
    end
    P[n+1, 1:n] .= 0.0
    P[n+2, 1:n] .= 0.0

    P[n+1, n+1] = 1.0
    P[n+2, n+2] = 1.0
    return P
end

function solve_gmres(μ̄, returns=asset_returns; return_lambda=false, init_weights=nothing, verbose=false)
    A, b, n = get_system(μ̄, returns, init_weights)
    P = precond_gmres(A)
    P⁻¹ = inv(P)
    PA = P⁻¹ * A
    Pb = P⁻¹ * b

    x, ch = gmres(PA, Pb, log=true)
    if ch.isconverged && verbose
        println("GMRES converged in ", ch.iters, " iterations.")
    elseif !ch.isconverged
        println("GMRES did not converge.")
    end
    if sum(x[1:n]) ≈ 1.0 == false
        @info "Weights sum to 1.0"
    end
    if !return_lambda
        return x[1:n]
    end
    return x
end

function rel_resid_norm(A, x, b)
    r = A * x .- b
    return norm(r) / norm(b)
end



μ̄_test = 0.1
A, b, n = get_system(μ̄_test, asset_returns)

println("\nBS Info:")
weights_bs = @time solve_backslash(μ̄_test, asset_returns; return_lambda=true)
sum_w_bs = sum(weights_bs[1:n])
μ_bs = dot(mean.(eachcol(asset_returns)), weights_bs[1:n])
rresid_bs = rel_resid_norm(A, weights_bs, b)

println("\nCG Info:")
weights_cg = @time solve_cg(μ̄_test, asset_returns; return_lambda=true, verbose=true)
sum_w_cg = sum(weights_cg[1:n])
μ_cg = dot(mean.(eachcol(asset_returns)), weights_cg[1:n])
rresid_cg = rel_resid_norm(A, weights_cg, b)

println("\nGMRES Info:")
weights_gmres = @time solve_gmres(μ̄_test, asset_returns; return_lambda=true, verbose=true)
sum_w_gmres = sum(weights_gmres[1:n])
μ_gmres = dot(mean.(eachcol(asset_returns)), weights_gmres[1:n])
rresid_gmres = rel_resid_norm(A, weights_gmres, b)

df_sol = DataFrame(
    Method = ["BS", "CG", "GMRES"],
    RRN = [rresid_bs, rresid_cg, rresid_gmres],
    Σw = [sum_w_bs, sum_w_cg, sum_w_gmres],
    μ = [μ_bs, μ_cg, μ_gmres],
)

sort!(df_sol, :RRN)
CSV.write(joinpath(@__DIR__, "tabular_output", "p4_solutions.csv"), df_sol)

# Portfolio variance from the min(RRN) solution (BS)
function get_pvar(weights, returns=asset_returns)
    return weights' * cov(Matrix(returns)) * weights
end
σ²ₚ = get_pvar(weights_bs[1:n], asset_returns)
println("\nPortfolio Variance (from BS solution): ", σ²ₚ)

# Weights from the min(RRN) solution (BS)
idx = ["Asset_$(i)" for i in 1:n]
df_weights_BS = DataFrame(Asset = idx, Weights = weights_bs[1:n])
df_weights_CG = DataFrame(Asset = idx, Weights = weights_cg[1:n])
df_weights_GMRES = DataFrame(Asset = idx, Weights = weights_gmres[1:n])

test_μ̄ = .01:(.1-.01)/50:.1 # 50 equidistance steps where  μ̄ ∈ [0.01, 0.1]

# using BS as it provides the most accurate solution
solutions_BS = [solve_backslash(test_μ̄[1]; return_lambda=false)]
σ²ₚ_list_BS = [get_pvar(solutions_BS[1])]

##
# solutions_CG = [solve_cg(test_μ̄[1]; return_lambda=false)]
# σ²ₚ_list_CG = [get_pvar(solutions_CG[1])]

# solutions_GMRES = [solve_gmres(test_μ̄[1]; return_lambda=false)]
# σ²ₚ_list_GMRES = [get_pvar(solutions_GMRES[1])]
for μ̄ in test_μ̄[2:end]
    w_bs = solve_backslash(μ̄; 
    return_lambda=false, 
    init_weights=solutions_BS[end])
    push!(solutions_BS, w_bs)
    push!(σ²ₚ_list_BS, get_pvar(w_bs))

    ##
    # w_cg = solve_cg(μ̄; 
    # return_lambda=false, 
    # init_weights=solutions_CG[end])
    # push!(solutions_CG, w_cg)
    # push!(σ²ₚ_list_CG, get_pvar(w_cg))

    # w_gmres = solve_gmres(μ̄; 
    # return_lambda=false, 
    # init_weights=solutions_GMRES[end])
    # push!(solutions_GMRES, w_gmres)
    # push!(σ²ₚ_list_GMRES, get_pvar(w_gmres))
end

df_sol_test_BS = DataFrame(
    μ̄ = test_μ̄,
    σ²ₚ = σ²ₚ_list_BS,
    σₚ = sqrt.(σ²ₚ_list_BS)
)

##
# df_sol_test_CG = DataFrame(
#     μ̄ = test_μ̄,
#     σ²ₚ = σ²ₚ_list_CG,
#     σₚ = sqrt.(σ²ₚ_list_CG)
# )

# df_sol_test_GMRES = DataFrame(
#     μ̄ = test_μ̄,
#     σ²ₚ = σ²ₚ_list_GMRES,
#     σₚ = sqrt.(σ²ₚ_list_GMRES)
# )


# Efficient frontier plot

@suppress_err begin # MatPlotLib warning about color mapping
    p = scatter(
        df_sol_test_BS.σₚ,
        df_sol_test_BS.μ̄;
        xlabel=L"\sigma_p",
        ylabel=L"\bar{\mu}",
        title="Efficient Frontier [BS]",
        legend=false,
    )
    savefig(joinpath(@__DIR__, "figure", "p4_efficient_frontier_BS.png"))

    ##
    # p = scatter(
    #     df_sol_test_CG.σₚ,
    #     df_sol_test_CG.μ̄;
    #     xlabel=L"\sigma_p",
    #     ylabel=L"\bar{\mu}",
    #     title="Efficient Frontier [CG]",
    #     legend=false,
    # )
    # savefig(joinpath(@__DIR__, "figure", "problem4_efficient_frontier_CG.png"))

    # p = scatter(
    #     df_sol_test_GMRES.σₚ,
    #     df_sol_test_GMRES.μ̄;
    #     xlabel=L"\sigma_p",
    #     ylabel=L"\bar{\mu}",
    #     title="Efficient Frontier [GMRES]",
    #     legend=false,
    # )
    savefig(joinpath(@__DIR__, "figure", "problem4_efficient_frontier_GMRES.png"))

end

CSV.write(joinpath(@__DIR__, "tabular_output", "p4_solutions.csv"), df_sol)
CSV.write(joinpath(@__DIR__, "tabular_output", "p4_test_solutions_BS.csv"), df_sol_test_BS)
CSV.write(joinpath(@__DIR__, "tabular_output", "p4_weights_BS.csv"), df_weights_BS)
# CSV.write(joinpath(@__DIR__, "tabular_output", "p4_weights_CG.csv"), df_weights_CG)
# CSV.write(joinpath(@__DIR__, "tabular_output", "p4_weights_GMRES.csv"), df_weights_GMRES)