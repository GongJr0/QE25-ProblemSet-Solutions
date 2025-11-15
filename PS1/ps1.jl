using Random, Statistics, Distributions,Plots,DataFrames,LaTeXStrings

#Problem 1

function clt(n, λ=1, r=1000)
    standardized = Float64[]
    for i in 1:r
        sample = rand(Poisson(λ), n)
        sm = mean(sample)
        z = (sm - λ) / (sqrt(λ) / sqrt(n))
        push!(standardized, z)
    end
    return standardized
end

function plot_clt()
    ns = [5, 25, 100, 1000]
    p = []           
    for n in ns
        data = clt(n)
        h = histogram(data,
            normalize=true,
            bins=20,
            title="N = $n",
            xlabel="z",
            ylabel="Frequency",
            legend=false,
            alpha=0.6)
        plot!(h, -4:0.01:4, pdf.(Normal(0,1), -4:0.01:4),
            color=:red, lw=2)
        xlims!(-4,4)
        ylims!(0,1)
        push!(p, h)
    end
    plot(p..., layout=(2,2), size=(800,600)) # dots used in same manner *args in python
end

plot_clt()

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

plot(alfas, p1_02, label="σ=0.2", xlabel=L"\alpha", ylabel="Equilibrium Price p1", title="Equilibrium Price vs α", legend=:topright)
savefig("equilibrium_price_sigma_02.png")
plot(alfas,c1_1s_02, label="Agent 1", xlabel=L"\alpha", ylabel="Consumption of Good 1", title="Consumption of Good 1 vs α (σ=0.2)", legend=:topright)
savefig("consumption_good1_agent1_sigma_02.png")
plot(alfas,c1_2s_02, label="Agent 2", xlabel=L"\alpha", ylabel="Consumption of Good 1", title="Consumption of Good 1 vs α (σ=0.2)", legend=:topright)
savefig("consumption_good1_agent2_sigma_02.png")
plot(alfas, p1_5, label="σ=5.0", xlabel=L"\alpha", ylabel="Equilibrium Price p1", title="Equilibrium Price vs α", legend=:topright)
savefig("equilibrium_price_sigma_5.png")
plot(alfas,c1_1s_5, label="Agent 1", xlabel=L"\alpha", ylabel="Consumption of Good 1", title="Consumption of Good 1 vs α (σ=5.0)", legend=:topright)
savefig("consumption_good1_agent1_sigma_5.png")
plot(alfas,c1_2s_5, label="Agent 2", xlabel=L"\alpha", ylabel="Consumption of Good 1", title="Consumption of Good 1 vs α (σ=5.0)", legend=:topright)
savefig("consumption_good1_agent2_sigma_5.png")

plot(p1, p2,
     layout = (1,2),
     size = (750,450))

savefig("equilibrium_prices_comparison.png")
   
eq_both_goods = df_02[c1_1s_02 .== c1_2s_02 .&& c2_1s_02 .== c2_2s_02, :] #there is no such alpha

#Elasticity affects the substituiability between goods. Low elasticity (σ=0.2) means goods are less substitutable, leading to more stable prices and consumption patterns as α changes. High elasticity (σ=5.0) means goods are highly substitutable, causing more significant fluctuations in prices and consumption as α varies.

















