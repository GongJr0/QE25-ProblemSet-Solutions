using QuadGK, Distributions, NLsolve, Plots,Parameters,LaTeXStrings
pyplot()

#Problem 3

# Parameters struct
@with_kw mutable struct ModelParams
    β::Float64 = 0.96
    α::Float64 = 0.33
    A::Float64 = 1.0
    δ::Float64 = 0.1
    γ::Float64 = 1.0
    k0::Float64 = 0.0
end

# Steady state computation
function compute_steady_state(β, α, A, δ)
    k_ss = ((1/β - 1 + δ) / (α * A))^(1/(α - 1))
    c_ss = A * k_ss^α - δ * k_ss
    i_ss = δ * k_ss
    y_ss = A * k_ss^α
    return k_ss, c_ss, i_ss, y_ss
end

# System of transition equations
function transition_equations!(F, x, params, T, c_ss, k_ss)
    @unpack β, α, A, δ, γ, k0 = params
    
    # Extract variables: [c_0, c_1, ..., c_T, k_1, k_2, ..., k_T]
    c = x[1:T+1]
    k = x[T+2:end]
    
    # Euler equations: t = 0, ..., T-1
    for t in 0:T-1
        k_next = k[t+1] 
        F[t+1] = c[t+1]^(-γ) - β * c[t+2]^(-γ) * (α * A * k_next^(α-1) + (1 - δ))
    end
    
    # Capital accumulation: t = 0, ..., T-1
    for t in 0:T-1
        k_current = (t == 0) ? k0 : k[t]
        k_next = k[t+1]
        y_current = A * k_current^α
        F[T + t + 1] = k_next - ((1 - δ) * k_current + y_current - c[t+1])
    end
    
    # Terminal condition: c_T = c_ss
    F[2*T + 1] = c[T+1] - c_ss
end

# Solve the transition path
function solve_transition(params, T, c_ss, k_ss)
    # Initial guess (as suggested in problem set)
    c_init = fill(c_ss, T+1)  # constant at steady state
    k_init = [params.k0 + t/T * (k_ss - params.k0) for t in 1:T]  # linear interpolation
    x0 = vcat(c_init, k_init)
    
    # Define the residual function
    function residuals!(F, x)
        transition_equations!(F, x, params, T, c_ss, k_ss)
    end
    
    # Solve the system
    sol = nlsolve(residuals!, x0, ftol=1e-10, method=:trust_region, show_trace=false)
    
    # Extract solution
    c_path = sol.zero[1:T+1]
    k_path = vcat(params.k0, sol.zero[T+2:end])
    
    # Check terminal condition
    terminal_error = abs(c_path[end] - c_ss) / c_ss
    
    # Compute final residuals
    F_final = zeros(2*T + 1)
    residuals!(F_final, sol.zero)
    
    return (
        c_path = c_path,
        k_path = k_path,
        converged = sol.f_converged,
        terminal_error = terminal_error,
        max_residual = maximum(abs.(F_final)),
        iterations = sol.iterations
    )
end


# Set parameters
β = 0.96
α = 0.33
A = 1.0
δ = 0.1

#1 : Compute steady state
k_ss, c_ss, i_ss, y_ss = compute_steady_state(β, α, A, δ)


println("Steady State")
println("\nk* = ", round(k_ss, digits=4))
println("\nc* = ", round(c_ss, digits=4))
println("\ni* = ", round(i_ss, digits=4))
println("\ny* = ", round(y_ss, digits=4))
println()

#  Solve transition for γ = 0.5
println("\nSolving for γ = 0.5")

T = 100
params_05 = ModelParams(β=β, α=α, A=A, δ=δ, γ=0.5, k0=0.5*k_ss)

println("\nStarting with T = $T periods...")
result_05 = solve_transition(params_05, T, c_ss, k_ss)

println("\nConverged: ", result_05.converged)
println("\nTerminal error (|c_T - c*|/c*): ", round(result_05.terminal_error * 100, digits=4), "%")
println("\nMax residual: ", round(result_05.max_residual, sigdigits=3))
println("\nIterations: ", result_05.iterations)

# Check if T is sufficient
if result_05.terminal_error > 0.001
    println("\nWarning: c_T is not close enough to c*. Consider increasing T.")
else
    println("\nTerminal condition satisfied (within 0.1%)")
end


# Step 3: Solve transition for γ = 2.0

println("\nSolving for γ = 2.0")

params_20 = ModelParams(β=β, α=α, A=A, δ=δ, γ=2.0, k0=0.5*k_ss)

println("\nStarting with T = $T periods...")
result_20 = solve_transition(params_20, T, c_ss, k_ss)

println("\nConverged: ", result_20.converged)
println("\nTerminal error (|c_T - c*|/c*): ", round(result_20.terminal_error * 100, digits=4), "%")
println("\nMax residual: ", round(result_20.max_residual, sigdigits=3))
println("\nIterations: ", result_20.iterations)

if result_20.terminal_error > 0.001
    println("\nWarning: c_T is not close enough to c*. Try increasing T (e.g., 150 or 200).")
else
    println("\nTerminal condition satisfied (within 0.1%)")
end


# Step 4: Compute derived variables for plotting
y_05 = A .* result_05.k_path.^α
y_20 = A .* result_20.k_path.^α
i_05 = y_05 .- result_05.c_path
i_20 = y_20 .- result_20.c_path

c_rate_05 = result_05.c_path ./ y_05
c_rate_20 = result_20.c_path ./ y_20
i_rate_05 = i_05 ./ y_05
i_rate_20 = i_20 ./ y_20

steady_c_rate = c_ss / y_ss
steady_i_rate = i_ss / y_ss

plots_ks = []  # Changed from plot_ks to plots_ks

ks_05 = plot(
    0:T, result_05.k_path,
    xlabel = "Time",
    ylabel = L"k_t",
    title = L"Capital Stock Path ($\gamma = 0.5$)",
    titlefontsize = 11,
    label = L"k_t",
    lw = 2,
    color = :blue
)
hline!(ks_05, [k_ss], 
    label = "Steady state k̄ = $(round(k_ss, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)

ks_20 = plot(
    0:T, result_20.k_path,
    xlabel = "Time",
    ylabel = L"k_t",
    title = L"Capital Stock Path ($\gamma = 2.0$)",
    titlefontsize = 11,
    label = L"k_t",
    lw = 2,
    color = :red
)
hline!(ks_20, [k_ss],
    label = "Steady state k̄ = $(round(k_ss, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)

push!(plots_ks, ks_05)
push!(plots_ks, ks_20)

plot(plots_ks..., 
    layout = (1, 2), 
    size=(1000, 400), 
    plot_title="Capital Stock Transition Paths for Different " * L"\gamma", 
    plot_titlefontsize=14,
)
savefig(joinpath(@__DIR__, "figure", "capital_stock.png"))

plots_crates = []
cr_05 = plot(
    0:T, c_rate_05,
    xlabel = "Time",
    ylabel = L"c_t / y_t",
    title = L"Consumption Rate Path ($\gamma = 0.5$)",
    titlefontsize = 11,
    label = L"c_t / y_t",
    lw = 2,
    color = :blue
)
hline!(cr_05, [steady_c_rate], 
    label = "Steady state c̄/ȳ = $(round(steady_c_rate, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)   
cr_20 = plot(
    0:T, c_rate_20,
    xlabel = "Time",
    ylabel = L"c_t / y_t",
    title = L"Consumption Rate Path ($\gamma = 2.0$)",
    titlefontsize = 11,
    label = L"c_t / y_t",
    lw = 2,
    color = :red
)
hline!(cr_20, [steady_c_rate],
    label = "Steady state c̄/ȳ = $(round(steady_c_rate, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)
push!(plots_crates, cr_05)
push!(plots_crates, cr_20)
plot(plots_crates..., 
    layout = (1, 2), 
    size=(1000, 400), 
    plot_title="Consumption Rate Transition Paths for Different " * L"\gamma", 
    plot_titlefontsize=14,
)
savefig(joinpath(@__DIR__, "figure", "consumption_rate.png"))

plots_irates = []
ir_05 = plot(
    0:T, i_rate_05,
    xlabel = "Time",
    ylabel = L"i_t / y_t",
    title = L"Investment Rate Path ($\gamma = 0.5$)",
    titlefontsize = 11,
    label = L"i_t / y_t",
    lw = 2,
    color = :blue
)
hline!(ir_05, [steady_i_rate], 
    label = "Steady state ī/ȳ = $(round(steady_i_rate, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)   
ir_20 = plot(
    0:T, i_rate_20,
    xlabel = "Time",
    ylabel = L"i_t / y_t",
    title = L"Investment Rate Path ($\gamma = 2.0$)",
    titlefontsize = 11,
    label = L"i_t / y_t",
    lw = 2,
    color = :red
)
hline!(ir_20, [steady_i_rate],
    label = "Steady state ī/ȳ = $(round(steady_i_rate, digits=3))",  
    ls = :dash, 
    color = :black, 
    lw = 1.5
)
push!(plots_irates, ir_05)
push!(plots_irates, ir_20)
plot(plots_irates..., 
    layout = (1, 2), 
    size=(1000, 400), 
    plot_title="Investment Rate Transition Paths for Different " * L"\gamma", 
    plot_titlefontsize=14,
)
savefig(joinpath(@__DIR__, "figure", "investment_rate.png"))


# Low γ means faster convergence to steady state. Household with low γ in our case γ=0.5 would like to consume and invest more today comparing to future, so they adjust their consumption and investment faster. On the other hand high γ means slower convergence to steady state. Household with high γ in our case γ=2.0 would like to smooth consumption more over time, so they adjust their consumption and investment slower.


