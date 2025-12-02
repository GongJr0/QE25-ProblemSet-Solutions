using NLsolve, QuadGK, Random, DataFrames, PrettyTables, Printf, Plots
Random.seed!(0)
pyplot()

# ============== Function Definitions ==============

# Portfolio Return
function Rp(ω::Float64, r::Float64, Rf::Float64)::Float64
    return ω*r + (1-ω)*Rf
end

# Wealth Utility Function
function u(W::Float64, γ::Float64)::Float64
    return W^(1-γ)/(1-γ)
end

function u′(W::Float64, γ::Float64)::Float64
    return W^(-γ)
end

# Parametrized Log-Normal PDF
function f_r(r::Float64, μ::Float64, σ::Float64)::Float64
    return 1/(r*σ*sqrt(2*pi)) * exp(-(log(r) - μ)^2 / (2*σ^2))
end

function foc_integral(
    ω̂::Vector{Float64},
    Rf::Float64,
    W::Float64,
    u′::Function,
    f_r::Function;
    atol::Float64 = 1e-8
)::Float64
    if length(ω̂) != 1
        error("ω̂ must be a scalar wrapped in a vector.")
    end

    integrand(r) = (r - Rf) *
                   u′(W * Rp(ω̂[1], r, Rf)) *
                   f_r(r)

    integral, err = quadgk(integrand, 0.0, Inf; atol=atol)
    if err > atol
        @warn "Integral did not converge within tolerance. err = $err"
    end
    return integral
end

function optimal_portfolio(
    W::Float64,
    Rf::Float64,
    γ::Float64,
    μ::Float64,
    σ::Float64;
    atol::Float64 = 1e-8
)

    u′_func(W) = u′(W, γ)
    f_r_func(r) = f_r(r, μ, σ)

    F(ω̂) = foc_integral(ω̂, Rf, W, u′_func, f_r_func; atol=atol)

    # Asserting ω̂ ∈ [0, 1] due to DomainError when leveraging/shorting ({ω̂ > 1} ∪ {ω̂ < 0}) is allowed

    F0 = F([0.0])
    F1 = F([1.0])

    if F0 * F1 > 0
        ω̂_opt = max(sign(F1), 0.0)
    else
        sol = nlsolve(
        F,
        [0.5])
        ω̂_opt = sol.zero[1]
    end
    return ω̂_opt
end

# ============== Solution Script ==============


# Test foc_integral with γ=0, W=1
γ = 0.0
W = 1.0

# Risk-free gross return in a plausible range
Rf_arr = collect(0.95:0.01:1.10)

μ_test = 0.0
σ_test = 0.05

u′_func(W) = u′(W, γ)

f_r_func(r) = f_r(r, μ_test, σ_test)

W_samples      = Float64[]
ω̂_samples     = Float64[]
Rf_samples     = Float64[]
integrals      = Float64[]
actuals        = Float64[]
errors         = Float64[]

for _ in 1:100
    ω̂ = rand()
    Rf = rand(Rf_arr)

    integral = foc_integral(ω̂, Rf, W, u′_func, f_r_func; atol=1e-8)
    actual   = exp(μ_test + σ_test^2 / 2) - Rf

    error = abs(integral - actual)

    push!(ω̂_samples, ω̂)
    push!(W_samples, W)
    push!(Rf_samples, Rf)
    push!(integrals, integral)
    push!(actuals, actual)
    push!(errors, error)
end

df = DataFrame(
    omega = ω̂_samples,
    Rf = Rf_samples,
    Integral = integrals,
    Actual = actuals,
    Error = errors,
    in_tol = errors .<= 1e-8
)

println("FOC Integral Test Results (γ=0, W=1):")
pretty_table(df)

ω̂_opt = optimal_portfolio(1.0, 1.02, 3.0, 0.05, 0.1; atol=1e-8)
println(@sprintf("Optimal portfolio weight ω̂*: %.6f", ω̂_opt))

# Plot ω̂ as a function of γ
γ_arr = collect(0.9:0.1:10.0)
ω̂_opt_arr = Float64[]
for γ in γ_arr
    ω̂_opt = optimal_portfolio(1.0, 1.02, γ, 0.05, 0.1; atol=1e-8)
    push!(ω̂_opt_arr, ω̂_opt)
end

plot(
    γ_arr,
    ω̂_opt_arr,
    xlabel = "Risk Aversion (γ)",
    ylabel = "Optimal Portfolio Weight (ω̂)",
    title = "Optimal Portfolio Weight vs Risk Aversion",
    legend = false,
    grid = true
)

out = joinpath(@__DIR__, "figure", "share_v_risk_aversion.png")
savefig(out)