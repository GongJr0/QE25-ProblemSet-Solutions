using QuadGK, Distributions, NLsolve, Plots

using Parameters

function foc_integral(ω, W, Rf, γ, μ, σ)
    dist = LogNormal(μ, σ)
    
    function integrand(r)
        Rp = ω * r + (1 - ω) * Rf
        W_Rp = W * Rp
        
        # Check for domain issues
        if W_Rp <= 0 || !isfinite(W_Rp)
            return 0.0
        end
        
        term = (r - Rf) * W_Rp^(-γ) * pdf(dist, r)
        
        # Check if result is finite
        return isfinite(term) ? term : 0.0
    end
    
    result, _ = quadgk(integrand, 0, Inf, rtol=1e-8)
    return result
end

# Quick check with γ = 0 
r = foc_integral(0.5, 1.0, 1.02, 0.0, 0.05, 0.2)
println("Integral result: ", r)

μ = 0.05
σ = 0.2
Rf = 1.02
E_R = exp(μ + σ^2/2)
println("E[R] = ", E_R)
println("E[R] - Rf = ", E_R - Rf)

function optimal_portfolio(W, Rf, γ, μ, σ; ω0=0.5)
    function foc!(F, x)
        # Clamp omega to reasonable bounds
        ω_val = clamp(x[1], -2.0, 5.0)
        F[1] = foc_integral(ω_val, W, Rf, γ, μ, σ)
    end
    
    sol = nlsolve(foc!, [ω0], ftol=1e-6, iterations=1000)
    
    if !converged(sol)
        println("Warning: γ=$γ did not converge")
        return NaN
    end
    
    return sol.zero[1]
end

# Test single value first
println("\nTesting single value:")
ω_star = optimal_portfolio(1.0, 1.02, 3.0, 0.05, 0.1)
println("ω* for γ=3: ", ω_star)

# Loop with error handling
w_stars = Float64[]
gammas = Float64[]

println("\nComputing portfolio shares...")
for γ in 0.1:0.01:10.0
    try
        ω_star = optimal_portfolio(1.0, 1.02, γ, 0.05, 0.2)
        
        if isfinite(ω_star)
            push!(w_stars, ω_star)
            push!(gammas, γ)
        end
        
        if abs(γ - round(γ)) < 0.01
            println("γ = $(round(γ, digits=1)): ω* = $(round(ω_star, digits=4))")
        end
    catch e
        println("Error at γ=$γ: ", e)
    end
end

println("\nPlotting $(length(gammas)) points...")

plot(gammas, w_stars, 
     xlabel="Risk Aversion (γ)", 
     ylabel="Optimal Risky Asset Share (ω*)", 
     title="Optimal Portfolio Share vs Risk Aversion", 
     legend=false,
     linewidth=2,
     marker=:circle,
     markersize=2)

#Problem 3 

using Parameters

# Step 1: Define parameters WITHOUT k0
@with_kw struct CalibrateParameters
    β::Float64 = 0.96
    α::Float64 = 0.33
    A::Float64 = 1.0
    δ::Float64 = 0.1
    k0::Float64 = 0.0  # Will be set later
end

# Step 2: Function to compute steady state
function compute_steady_state(β, α, A, δ)
    ks = ((1/β - 1 + δ) / (α * A))^(1/(α - 1))
    cs = A * ks^α - δ * ks
    is = δ * ks
    ys = A * ks^α
    return ks, cs, is, ys
end

# Step 3: Compute steady state
β, α, A, δ = 0.96, 0.33, 1.0, 0.1
ks, cs, is, ys = compute_steady_state(β, α, A, δ)

println("Steady State Values:")
println("Capital (k*): ", ks)
println("Consumption (c*): ", cs)
println("Investment (i*): ", is)
println("Output (y*): ", ys)

# Step 4: Create params with k0 = 0.5 * ks
params = CalibrateParameters(β=β, α=α, A=A, δ=δ, k0=0.5*ks)

println("\nInitial capital (k0): ", params.k0)

function transition_equations!(F, x, params, T, cs, ks)
    @unpack β, α, A, δ, k0 = params
    
    # x = [c0, c1, ..., cT, k1, k2, ..., kT]
    # Length: (T+1) + T = 2T+1
    c = x[1:T+1]              # c[1]=c0, c[2]=c1, ..., c[T+1]=cT
    k_vec = x[T+2:end]        # k[1]=k1, k[2]=k2, ..., k[T]=kT
    
    # Euler equations for t = 0, ..., T-1 (T equations)
    for t in 0:T-1
        c_t = c[t+1]          # consumption at time t
        c_tp1 = c[t+2]        # consumption at time t+1
        
        # Capital at time t+1:
        # if t=0: k_{t+1} = k1 = k_vec[1]
        # if t=T-1: k_{t+1} = kT = k_vec[T]
         k_t = (t == 0) ? k0 : k_vec[t]
        k_tp1 = k_vec[t+1]
        
        F[t+1] = c_t^(-γ) - β * c_tp1^(-γ) * (α * A * k_tp1^(α - 1) + (1 - δ))
    end
    
    # Capital accumulation for t = 0, ..., T-1 (T equations)
    for t in 0:T-1
        c_t = c[t+1]          # consumption at time t
        
        # Capital at time t:
        # if t=0: k_t = k0 (given)
        # if t≥1: k_t = k_{t} = k_vec[t]
        k_t = (t == 0) ? k0 : k_vec[t]
        
        # Capital at time t+1: k_{t+1} = k_vec[t+1]
        k_tp1 = k_vec[t+1]
        
        F[T + t + 1] = k_tp1 - ((1 - δ) * k_t + A * k_t^α - c_t)
    end
    
    # Terminal condition: cT = cs (1 equation)
    F[2*T + 1] = c[T+1] - cs
    
    return F
