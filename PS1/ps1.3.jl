using Random, Statistics, Distributions,Plots,DataFrames,LaTeXStrings,CSV,Distributions,LinearAlgebra,IterativeSolvers,PrettyTables
pyplot()

#Problem 3

# Marshallian demand for good 1
function marshallian_good1(p1, p2, αi, σi, ωi_1, ωi_2)
    n1 = αi^σi * p1^(-σi)
    d1 = αi^σi * p1^(1-σi) + (1 - αi)^σi * p2^(1-σi)
    i1 = p1 * ωi_1 + p2 * ωi_2
    return (n1/d1) * i1
end

# Marshallian demand for good 2
function marshallian_good2(p1, p2, αi, σi, ωi_1, ωi_2)
    n2 = (1 - αi)^σi * p2^(-σi)
    d2 = αi^σi * p1^(1-σi) + (1 - αi)^σi * p2^(1-σi)
    i2 = p1 * ωi_1 + p2 * ωi_2
    return (n2/d2) * i2
end

# Market clearing condition (excess demand for good 1)
function market_clear(p1, α, σ, ω)
    p2 = 1.0  # normalized
    
    c1_1 = marshallian_good1(p1, p2, α[1], σ[1], ω[1][1], ω[1][2])
    c2_1 = marshallian_good1(p1, p2, α[2], σ[2], ω[2][1], ω[2][2])
    
    total_demand = c1_1 + c2_1
    total_supply = ω[1][1] + ω[2][1]
    
    return total_demand - total_supply  # excess demand (should be zero)
end

function newton_method(α, σ, ω, p1_init=1.0, tol=1e-12, max_iter=100)
    p1 = p1_init
    h = 1e-8
    
    for i in 1:max_iter
        f = market_clear(p1, α, σ, ω)
        
        
        f_prime = (market_clear(p1 + h, α, σ, ω) - market_clear(p1 - h, α, σ, ω)) / (2*h)
        
    
        p1_new = p1 - f / f_prime
        
        # check market clearing and
        if abs(f) < tol && abs(p1_new - p1) < tol
            println("Converged in $i iterations")
            return p1_new
        end
        
        p1 = p1_new
    end
    
    println("Max iterations reached")
    return p1
end

r = 0.01:0.01:0.99
alfas= []

σ_02 = [.2, .2]
σ_5 = [5.0, 5.0]
ω = [(1.0, 1.0), (0.5, 1.5)]

p1_02 = []
p2_02 = []
c1_1s_02 = []
c1_2s_02 = []
c2_1s_02 = []
c2_2s_02 = []
demand1_02 = []
supply1_02 = []
demand2_02 = []
supply2_02 = []


for alf in r
    α = [alf, 1 - alf]
    p1_eq = newton_method(α, σ_02, ω)
    p2_eq = 1.0
    
    c1_1 = marshallian_good1(p1_eq, p2_eq, α[1], σ_02[1], ω[1][1], ω[1][2])
    c1_2 = marshallian_good2(p1_eq, p2_eq, α[1], σ_02[1], ω[1][1], ω[1][2])
    c2_1 = marshallian_good1(p1_eq, p2_eq, α[2], σ_02[2], ω[2][1], ω[2][2])
    c2_2 = marshallian_good2(p1_eq, p2_eq, α[2], σ_02[2], ω[2][1], ω[2][2])
    
    total_demand1 = c1_1 + c2_1
    total_demand2 = c1_2 + c2_2
    total_supply1 = ω[1][1] + ω[2][1]
    total_supply2 = ω[1][2] + ω[2][2]
    
    # Check if market clears
    if !isapprox(total_demand1, total_supply1, atol=1e-10) || !isapprox(total_demand2, total_supply2, atol=1e-10)
        println("Market not cleared at α=$alf")
        break
    end
    
    push!(alfas, alf)
    push!(p1_02, p1_eq)
    push!(p2_02, p2_eq)
    push!(demand1_02, total_demand1)
    push!(supply1_02, total_supply1)
    push!(demand2_02, total_demand2)
    push!(supply2_02, total_supply2)
    push!(c1_1s_02, c1_1)
    push!(c1_2s_02, c1_2)
    push!(c2_1s_02, c2_1)
    push!(c2_2s_02, c2_2)
end

# Create DataFrame from the arrays
df_02 = DataFrame(
    alpha = alfas,
    p1 = p1_02,
    p2 = p2_02,
    c1_1 = c1_1s_02,
    c1_2 = c1_2s_02,
    c2_1 = c2_1s_02,
    c2_2 = c2_2s_02,
    demand1 = demand1_02,
    supply1 = supply1_02,
    demand2 = demand2_02,
    supply2 = supply2_02
)

p1_5 = []
p2_5 = []
c1_1s_5 = []
c1_2s_5 = []
c2_1s_5 = []
c2_2s_5 = []
demand1_5 = []
supply1_5 = []
demand2_5 = []
supply2_5 = []

for alf in r
    α = [alf, 1 - alf]
    p1_eq = newton_method(α, σ_5, ω)
    p2_eq = 1.0
    
    c1_1 = marshallian_good1(p1_eq, p2_eq, α[1], σ_5[1], ω[1][1], ω[1][2])
    c1_2 = marshallian_good2(p1_eq, p2_eq, α[1], σ_5[1], ω[1][1], ω[1][2])
    c2_1 = marshallian_good1(p1_eq, p2_eq, α[2], σ_5[2], ω[2][1], ω[2][2])
    c2_2 = marshallian_good2(p1_eq, p2_eq, α[2], σ_5[2], ω[2][1], ω[2][2])
    
    total_demand1 = c1_1 + c2_1
    total_demand2 = c1_2 + c2_2
    total_supply1 = ω[1][1] + ω[2][1]
    total_supply2 = ω[1][2] + ω[2][2]
    
    # Check if market clears
    if !isapprox(total_demand1, total_supply1, atol=1e-10) || !isapprox(total_demand2, total_supply2, atol=1e-10)
        println("Market not cleared at α=$alf")
        break
    end
    
    push!(p1_5, p1_eq)
    push!(p2_5, p2_eq)
    push!(demand1_5, total_demand1)
    push!(supply1_5, total_supply1)
    push!(demand2_5, total_demand2)
    push!(supply2_5, total_supply2)
    push!(c1_1s_5, c1_1)
    push!(c1_2s_5, c1_2)
    push!(c2_1s_5, c2_1)
    push!(c2_2s_5, c2_2)
end

# Create DataFrame from the arrays
df_5 = DataFrame(
    alpha = alfas,
    p1 = p1_5,
    p2 = p2_5,
    c1_1 = c1_1s_5,
    c1_2 = c1_2s_5,
    c2_1 = c2_1s_5,
    c2_2 = c2_2s_5,
    demand1 = demand1_5,
    supply1 = supply1_5,
    demand2 = demand2_5,
    supply2 = supply2_5
)

print(df_02)
print(df_5)

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

savefig(joinpath(@__DIR__, "figure", "p3_price.png"))

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
savefig(joinpath(@__DIR__, "figures", "p3_consumption.png"))

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

eq_df = DataFrame(
    σ=[0.2, 5.0],
    α1=[eq_02.α1, eq_5.α1],
    c1_1=[eq_02.c1_1, eq_5.c1_1],
    c2_1=[eq_02.c2_1, eq_5.c2_1],
    diff=[eq_02.diff, eq_5.diff]
)

# Complete DataFrames and Rows satisfying equilibrium condition to CSVs
out_df02 = joinpath(@__DIR__, "tabular_output", "p3_df_02.csv")
CSV.write(out_df02, df_02)

out_df5 = joinpath(@__DIR__, "tabular_output", "p3_df_5.csv")
CSV.write(out_df5, df_5)

out_eq = joinpath(@__DIR__, "tabular_output", "p3_equilibria.csv")
CSV.write(out_eq, eq_df)

#Elasticity affects the substituiability between goods. Low elasticity (σ=0.2) means goods are less substitutable, leading to more stable prices and consumption patterns as α changes. High elasticity (σ=5.0) means goods are highly substitutable, causing more significant fluctuations in prices and consumption as α varies.