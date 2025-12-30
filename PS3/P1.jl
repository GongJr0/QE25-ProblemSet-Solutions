using LinearAlgebra, Plots, LaTeXStrings

Z_mat = [
    0.6 0.3 0.1;
    0.2 0.6 0.2;
    0.1 0.3 0.6
]

function X_pol(Zt::Int, Xt::Int)
    if Zt == 1
        return 0
    elseif Zt == 2
        return Xt
    elseif Zt == 3 && Xt <= 4
        return min(Xt + 1, 5)
    elseif Zt == 3 && Xt == 5
        return 3
    end
end

# Joint transition matrix
Q = zeros(18,18)

for Xt in 0:5
    for Zt in 1:3
        X_next = X_pol(Zt, Xt)
        for Z_next in 1:3
            row = Xt * 3 + Zt
            col = X_next * 3 + Z_next
            Q[row, col] = Z_mat[Zt, Z_next]
        end
    end
end

rowsums = []
for i in 1:18
    push!(rowsums, sum(Q[i, :]))
end
println("Rows sum to 1: ", all(x -> isapprox(x, 1.0), rowsums))

# Stationary distribution (6x3 joint matrix; not flattened)
I_mat = I(18)

A = vcat((Q' - I_mat), ones(1, 18))
b = vcat(zeros(18), [1.0])

π = vec(A \ b)

Ψ_star = zeros(6, 3)

for Xt in 0:5
    for Zt in 1:3
        row = Xt * 3 + Zt
        Ψ_star[Xt + 1, Zt] = π[row]
    end
end

println("Stationary distribution sums to 1: ", isapprox(sum(Ψ_star), 1.0))
println("π = Qᵀ * π: ", isapprox(π, Q' * π)) # π is the flattened Ψ_star.

println("Stationary distribution Ψ*: ")
println(Ψ_star)
println()

marginal_X = sum(Ψ_star, dims=2)  # Row sums
marginal_Z = sum(Ψ_star, dims=1)  # Column sums

println("Marginal of X sums to 1: ", isapprox(sum(marginal_X), 1.0))
println("Marginal distribution of X: ")
println(marginal_X)
println()
println("Marginal of Z sums to 1: ", isapprox(sum(marginal_Z), 1.0))
println("Marginal distribution of Z: ")
println(marginal_Z)

# E[X]
E_X = sum((0:5) .* vec(marginal_X))
println("E[X]: ", E_X)

# E[X | Z=z]
E_X_given_Z = zeros(3)
for z in 1:3
    marginal_X_given_Z = Ψ_star[:, z] / marginal_Z[z]
    E_X_given_Z[z] = sum((0:5) .* vec(marginal_X_given_Z))
    println("E[X | Z=$z]: ", E_X_given_Z[z])
end

# Plots
p = plot(
    (0:5),
    vec(marginal_X), 
    title = L"\Psi^*_X", 
    xlabel="X", 
    ylabel="P(X)", 
    legend=false
)
savefig(p, "PS3/figure/p1_marginal_X.png")

p2 = plot(
    (1:3),
    vec(marginal_Z), 
    title = L"\Psi^*_Z", 
    xlabel="Z", 
    ylabel="P(Z)", 
    legend=false
)
savefig(p2, "PS3/figure/p1_marginal_Z.png")

p3 = plot(
    (1:3),
    E_X_given_Z,
    title = L"E[X | Z=z]",
    xlabel = "Z",
    ylabel = "E[X | Z=z]",
    legend = false
)
savefig(p3, "PS3/figure/p1_E_X_given_Z.png")