using QuadGK
using Distributions
using NLsolve
using Plots

function foc_integral(ω, W, Rf, γ, μ, σ)
    dist = LogNormal(μ, σ)
    
    function integrand(r)
        r_p = ω*r + (1 - ω)*Rf
        
        if r_p <= 0
            return 0.0
        end
        
        return (r - Rf) * (W * r_p)^(-γ) * pdf(dist, r)
    end
    
    # Integrate from 0 to infinity
    integral, err = quadgk(integrand, 0, Inf; rtol=1e-8)
    return integral
end

# Verify with γ = 0 (linear utility case)
result = foc_integral(0.5, 1.0, 1.02, 0.0, 0.05, 0.2)
println("Integral result: ", result)

μ = 0.05
σ = 0.2
Rf = 1.02
E_R = exp(μ + σ^2/2)
println("E[R] = ", E_R)
println("E[R] - Rf = ", E_R - Rf)


function optimal_portfolio(W, Rf, γ, μ, σ; ω_min=-0.5, ω_max=1.0)
    
    f = ω -> foc_integral(ω[1], W, Rf, γ, μ, σ)
    result = nlsolve(f, [0.5]; ftol=1e-8)
    ω_star = result.zero[1]
    
    # Apply constraints
    if ω_star < ω_min
        println("  γ=$(round(γ, digits=2)): ω*=$(round(ω_star, digits=3)) < $ω_min, constraining to $ω_min")
        return ω_min
    elseif ω_star > ω_max
        println("  γ=$(round(γ, digits=2)): ω*=$(round(ω_star, digits=3)) > $ω_max, constraining to $ω_max")
        return ω_max
    else
        return ω_star
    end
end

# Test scenario
scenario1 = optimal_portfolio(1.0, 1.02, 3.0, 0.05, 0.1)
println("\nOptimal share ω* for γ=3.0: ", round(scenario1, digits=6))
println()

γ_range = 0.1:0.1:10.0  

# Fixed parameters
W = 1.0
Rf = 1.02
μ = 0.05
σ = 0.1

# Set economic constraints
ω_min = -0.5  # Allow short selling (max 50% short of risky asset)
ω_max = 1.0   # No borrowing allowed! Max 100% in risky asset


ω_star_values = Float64[]

for γ in γ_range
    ω_star = optimal_portfolio(W, Rf, γ, μ, σ; ω_min=ω_min, ω_max=ω_max)
    push!(ω_star_values, ω_star)
end

# Create the plot
p = plot(γ_range, ω_star_values,
    xlabel = "Risk Aversion (γ)",
    ylabel = "Optimal Portfolio Share (ω*)",
    title = "Optimal Risky Asset Share vs. Risk Aversion",
    lw = 2.5,
    color = :blue,
    legend = :topright,
    label = "ω*",
    grid = true
)
savefig(joinpath(@__DIR__, "figure", "optimal_portfolio_share.png"))







