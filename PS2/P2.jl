using QuadGK, Distributions, NLsolve, Plots, Parameters, LaTeXStrings, Random, Statistics, Optim

function simulate_model(θ, T, σL, σH)
    ρ = θ[1]
    p = θ[2]
    
    logy = zeros(T)
    logy[1] = 0.0
    
    # Generate regime states
    x_t = [rand(Bernoulli(p)) for t in 1:T]
    
    # Generate shocks based on regime
    ε_t = zeros(T)
    for t in 1:T
        if x_t[t] == 1
            ε_t[t] = rand(Normal(0, σL))
        else
            ε_t[t] = rand(Normal(0, σH))
        end
    end
    
    # Generate log output
    for t in 1:T-1
        logy[t+1] = ρ * logy[t] + ε_t[t]
    end
    
    # Discard burn-in
    logy = logy[101:end]
    
    return logy
end

println("\nGenerating observed data...\n")
Random.seed!(2024)
obs_data = simulate_model((0.90, 0.80), 600, 0.1, 0.3)
println("\nObserved data: T = ", length(obs_data), " observations\n")


function smm_objective(θ, observed_data, σL, σH, S)
    # Set seed based on θ to ensure consistent simulations
    seed_value = Int(round(θ[1] * 10000)) + Int(round(θ[2] * 10000))
    Random.seed!(seed_value)
    
    # Observed moments
    m_1_obs = std(observed_data)
    m_2_obs = cor(observed_data[2:end], observed_data[1:end-1])
    Δlogy_obs = diff(observed_data)
    m_3_obs = kurtosis(Δlogy_obs)
    
    # Storage for S simulations
    m_1_sims = zeros(S)
    m_2_sims = zeros(S)
    m_3_sims = zeros(S)
    
    # Run S simulations with current θ
    for s in 1:S
        sim_data = simulate_model(θ, length(observed_data)+100, σL, σH)
        
        m_1_sims[s] = std(sim_data)
        m_2_sims[s] = cor(sim_data[2:end], sim_data[1:end-1])
        Δlogy_sim = diff(sim_data)
        m_3_sims[s] = kurtosis(Δlogy_sim)
    end
    
    # Average simulated moments
    m_1_sim_avg = mean(m_1_sims)
    m_2_sim_avg = mean(m_2_sims)
    m_3_sim_avg = mean(m_3_sims)
    
    # Compute objective
    obj = (m_1_obs - m_1_sim_avg)^2 + 
          (m_2_obs - m_2_sim_avg)^2 + 
          (m_3_obs - m_3_sim_avg)^2
    
    return obj
end

function optimize_smm(observed_data, σL, σH, S)
    # Initial guess
    θ0 = [0.85, 0.70]
    
    lower_bounds = [0.5, 0.5]
    upper_bounds = [0.99, 0.95]

    obj_fun = θ -> smm_objective(θ, observed_data, σL, σH, S)
    
    println("\nStarting optimization...\n")
    result = optimize(obj_fun, 
                      lower_bounds, 
                      upper_bounds, 
                      θ0, 
                      Fminbox(BFGS()))
    
    # Extract results
    θ_est = Optim.minimizer(result)
    
    println("\nSMM ESTIMATION RESULTS")
    println("\nEstimated Parameters:")
    println("'\nρ̂ = ", round(θ_est[1], digits=4))
    println("\np̂ = ", round(θ_est[2], digits=4))
    println("\nTrue Parameters:")
    println("\nρ = 0.9000")
    println("\np = 0.8000")
    println("\nEstimation Errors:")
    println("\nρ̂ - ρ = ", round(θ_est[1] - 0.90, digits=4))
    println("\np̂ - p = ", round(θ_est[2] - 0.80, digits=4))
    println("\nObjective Function:")
    println("\nQ(θ̂) = ", round(Optim.minimum(result), digits=6))
    println("\nOptimization Status:")
    println("\nConverged: ", Optim.converged(result))
    println("\nIterations: ", Optim.iterations(result))
    
    return θ_est, result
end

θ_estimated, opt_result = optimize_smm(obs_data, 0.1, 0.3, 100)

# Generate simulated data with T+100 to account for burn-in
sim_data = simulate_model(θ_estimated, 600, 0.1, 0.3)

# TIME SERIES PLOT - Overlayed
comparison_plot = plot(
    1:length(obs_data), obs_data,
    xlabel = "Time",
    ylabel = "log y_t",
    title = "Observed vs Simulated (Overlayed)",
    titlefontsize = 11,
    label = "Observed",
    lw = 2,
    color = :blue,
    alpha = 0.7
)
plot!(comparison_plot,
    1:length(sim_data), sim_data,
    label = "Simulated (θ̂)",
    lw = 2,
    color = :red,
    alpha = 0.7,
    linestyle = :dash
)

# Save time series plot
savefig(comparison_plot, joinpath(@__DIR__, "figure", "smm_output_ts_data.png"))

# HISTOGRAM PLOT - Overlayed
Δlogy_obs = diff(obs_data)
Δlogy_sim = diff(sim_data)

hist_comparison = histogram(
    Δlogy_obs,
    xlabel = "Δ log y_t",
    ylabel = "Density",
    title = "Distribution of Changes: Observed vs Simulated",
    titlefontsize = 11,
    label = "Observed",
    color = :blue,
    alpha = 0.6,
    bins = 30,
    normalize = :pdf
)
histogram!(hist_comparison,
    Δlogy_sim,
    label = "Simulated (θ̂)",
    color = :red,
    alpha = 0.6,
    bins = 30,
    normalize = :pdf
)

# Save histogram plot
savefig(hist_comparison, joinpath(@__DIR__, "figure", "smm_histogram_data.png"))
















