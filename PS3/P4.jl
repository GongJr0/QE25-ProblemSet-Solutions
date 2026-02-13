using Parameters,QuantEcon, LinearAlgebra, Plots, LaTeXStrings,Random, Statistics, StatsBase
pyplot()

@with_kw struct Params
    α::Float64 = 0.1
    f_bar::Float64 = 1.2
    δ::Float64 = 0.1
    γ::Float64 = 1.5
    β::Float64 = 0.95
    ρ::Float64 = 0.98
    σ::Float64 = 0.15
    ψ::Float64 = 0.5
end

#4.1 and 4.2

function f_h(h, params::Params)
    return min(h^params.α + 0.1, params.f_bar)
end

function h_next(h, e, params::Params)
    return max(h + e - params.δ, 0.0)
end

function utility(c, γ)
    if c <= 0
        return -Inf
    end
    return (c^(1 - γ)) / (1 - γ)
end

function flow_payoff(w, h, e, params::Params)
    c = w * f_h(h, params) * (1 - e)
    if c <= 0
        return -Inf
    end
    return utility(c, params.γ) - params.ψ * e
end

function wage_process(N, params::Params)
    mc = QuantEcon.rouwenhorst(N, params.ρ, params.σ)
    w_grid = exp.(mc.state_values)
    P_w = mc.p
    return w_grid, P_w
end

function set_up_grids(params::Params)
    h_max = 15 # set as 15 because according to the question descrption h' becomes flat around 13.8 the additional 1.2 is kinda buffer, thus we set the max to 15.
    h_grid = collect(0.0:0.01:h_max)
    e_grid = collect(0.0:0.01:1.0)
    w_grid, P_w = wage_process(7, params)
    return h_grid, e_grid, w_grid, P_w
end

function interpolation(h_prime, i_w, h_grid, V_old)
    if h_prime <= h_grid[1]
        return V_old[1, i_w]
    end
    
    if h_prime >= h_grid[end]
        return V_old[end, i_w]
    end
    
    idx = searchsortedfirst(h_grid, h_prime)
    
    h_low = h_grid[idx - 1]
    h_high = h_grid[idx]
    V_low = V_old[idx - 1, i_w]
    V_high = V_old[idx, i_w]
    
    slope = (V_high - V_low) / (h_high - h_low)
    V_interp = V_low + slope * (h_prime - h_low)
    
    return V_interp
end

function bellman_operator!(V_new, V_old, policy, h_grid, e_grid, w_grid, P_w, params)
    n_h = length(h_grid)  
    n_w = length(w_grid)
    
    for i_h in 1:n_h
        h = h_grid[i_h]
        
        for i_w in 1:n_w
            w = w_grid[i_w]
            
            max_value = -Inf
            best_e = 0.0     

            for e in e_grid
                current_payoff = flow_payoff(w, h, e, params)

                if current_payoff == -Inf
                    continue
                end

                h_prime = h_next(h, e, params)
                
                expected_continuation = 0.0
                
                for i_w_next in 1:n_w
                    prob = P_w[i_w, i_w_next]
                    V_next = interpolation(h_prime, i_w_next, h_grid, V_old)
                    expected_continuation += prob * V_next
                end

                total_value = current_payoff + params.β * expected_continuation
                
                if total_value > max_value
                    max_value = total_value
                    best_e = e
                end
            end 

            V_new[i_h, i_w] = max_value  
            policy[i_h, i_w] = best_e     
        end 
    end  
end

function vfi(params::Params; tol=1e-6, max_iter=1000, verbose=true)
    h_grid, e_grid, w_grid, P_w = set_up_grids(params)
    n_h = length(h_grid)
    n_w = length(w_grid)  
    
    V_old = zeros(n_h, n_w)    
    V_new = zeros(n_h, n_w)    
    policy = zeros(n_h, n_w)   
    
    if verbose
        println()
        println("\nVALUE FUNCTION ITERATION")
        println("\nGrid sizes:")
        println("\n  - Human capital: $(n_h) points from $(h_grid[1]) to $(h_grid[end])")
        println("\n  - Wages: $(n_w) states")
        println("\n  - Education: $(length(e_grid)) choices from 0 to 1")
        println("\n  - Total states: $(n_h * n_w)")
        println("\nStarting iterations...")
        println()
    end

    for iter in 1:max_iter
        bellman_operator!(V_new, V_old, policy, h_grid, e_grid, w_grid, P_w, params)

        diff = maximum(abs.(V_new - V_old))

        if verbose && (iter % 10 == 0)
            println("Iteration $iter: max difference = $(round(diff, digits=8))")
        end

        if diff < tol
            if verbose
                println("\nConverged in $iter iterations!")
                println("\nFinal max difference: $(round(diff, digits=10))")
                println("\nThis means V_new ≈ V_old (within tolerance $tol)")
                
            end
            return V_new, policy, h_grid, e_grid, w_grid, P_w, iter
        end
        
        V_old .= V_new
    end
    
    if verbose
        println()
        println("\nWarning: Did not converge in $max_iter iterations")
        println("\nFinal difference: $(maximum(abs.(V_new - V_old)))")
        println("\nYou may need to:")
        println("\n  - Increase max_iter")
        println("\n  - Check your parameter values")
        println("\n  - Reduce tolerance")
        println()
    end
    
    return V_new, policy, h_grid, e_grid, w_grid, P_w, max_iter
end

function simulate_path(rng, policy, h0, w0_idx, n_periods, h_grid, w_grid, P_w, params)

    h_path = zeros(n_periods)
    w_path = zeros(n_periods)
    e_path = zeros(n_periods)
    c_path = zeros(n_periods)

    h_path[1] = h0
    w_idx = w0_idx

    for t in 1:n_periods
        w_path[t] = w_grid[w_idx]
        h = h_path[t]

        e = interpolation(h, w_idx, h_grid, policy)
        e_path[t] = e

        c_path[t] = w_path[t] * f_h(h, params) * (1 - e)

        if t < n_periods
            h_path[t+1] = h_next(h, e, params)
            w_idx = sample(rng, 1:length(w_grid), Weights(P_w[w_idx, :]))
        end
    end

    return h_path, w_path, e_path, c_path
end


function simulate_multiple_paths(policy, h0, w0_idx, n_paths, n_periods,
                                 h_grid, w_grid, P_w, params; seed=2026)

    all_h_paths = []
    all_w_paths = []
    all_e_paths = []
    all_c_paths = []

    for i in 1:n_paths
        rng = MersenneTwister(seed + i - 1)

        h_path, w_path, e_path, c_path = simulate_path(
            rng, policy, h0, w0_idx, n_periods, h_grid, w_grid, P_w, params
        )

        push!(all_h_paths, h_path)
        push!(all_w_paths, w_path)
        push!(all_e_paths, e_path)
        push!(all_c_paths, c_path)
    end

    return all_h_paths, all_w_paths, all_e_paths, all_c_paths
end


#4.3
params = Params()
V, policy, h_grid, e_grid, w_grid, P_w, iters = vfi(params; verbose=true)


#4.4
w_indices = [1, 4, 7]                   
w_labels  = ["Low wage", "Medium wage", "High wage"]
colors    = [:blue, :green, :red]

println("Plotting for wage levels:")

for (i, idx) in enumerate(w_indices)
    println("  $(w_labels[i]) (w[$idx]) = $(round(w_grid[idx], digits=4))")
end
println()


p_education = plot(
    xlabel = L"Human\ Capital\ h",
    ylabel = L"Education\ effort\ e^*(h',w')",
    title = "Optimal Education Choice",
    titlefontsize = 11,
    legend = :topright,
    ylims = (0, 1),
    grid = true
)

for (i_w, label, color) in zip(w_indices, w_labels, colors)
    plot!(p_education, h_grid, policy[:, i_w],
        label = "$(label): w = $(round(w_grid[i_w], digits=3))",
        lw = 2,
        color = color
    )
end


savefig(p_education, joinpath(@__DIR__, "figure", "p4_education_plot.png"))


p_value = plot(
    xlabel = L"Human\ Capital\ h",
    ylabel = L"Value\ V(h',w')",
    title = "Value Function",
    titlefontsize = 11,
    legend = :bottomright,
    grid = true
)

for (i_w, label, color) in zip(w_indices, w_labels, colors)
    plot!(p_value, h_grid, V[:, i_w],
        label = "$(label): w = $(round(w_grid[i_w], digits=3))",
        lw = 2,
        color = color
    )
end


savefig(p_value, joinpath(@__DIR__, "figure", "p4_value_plot.png"))


p_consumption = plot(
    xlabel = L"Human\ Capital\ h",
    ylabel = L"Consumption\ c^*(h',w')",
    title = "Consumption",
    titlefontsize = 11,
    legend = :outerright,
    grid = true
)

for (i_w, label, color) in zip(w_indices, w_labels, colors)
    w = w_grid[i_w]
    consumption = [w * f_h(h, params) * (1 - policy[i_h, i_w])
                   for (i_h, h) in enumerate(h_grid)]

    plot!(p_consumption, h_grid, consumption,
        label = "$(label): w = $(round(w, digits=3))",
        lw = 2,
        color = color
    )
end

savefig(p_consumption, joinpath(@__DIR__, "figure", "p4_consumption_plot.png"))

#4.5
n_paths = 5
n_periods = 1100
burn_in = 100
h0 = 1.0
w0_idx = 4
seed = 7
println("\nSimulating paths...")
all_h, all_w, all_e, all_c = simulate_multiple_paths(
    policy, h0, w0_idx, n_paths, n_periods, h_grid, w_grid, P_w, params, seed=seed
)
println("\nSimulation complete")

start_idx = burn_in + 1
end_idx = burn_in + 100


colors = [:blue, :red, :green, :orange, :purple]


p_h = plot(
    title = "Human Capital Over Time",
    xlabel = L"Period\ (after\ burn-in)",
    ylabel = L"Human\ Capital\ h_t",
    legend = :bottomright,
    grid = true
)

for i in 1:n_paths
    periods = 1:100
    h_plot = all_h[i][start_idx:end_idx]
    plot!(p_h, periods, h_plot, 
          label = "Path $i", 
          color = colors[i], 
          lw = 1.5, 
          alpha = 0.8)
end

savefig(p_h, joinpath(@__DIR__, "figure", "p4_simulation_human_capital.png"))
println("Saved: p4_simulation_human_capital.png")


p_w = plot(
    title = "Wage Shocks Over Time",
    xlabel = L"Period\ (after\ burn-in)",
    ylabel = L"Wage\ w_t",
    legend = :topright,
    grid = true
)

for i in 1:n_paths
    periods = 1:100
    w_plot = all_w[i][start_idx:end_idx]
    plot!(p_w, periods, w_plot, 
          label = "Path $i", 
          color = colors[i], 
          lw = 1.5, 
          alpha = 0.8)
end

savefig(p_w, joinpath(@__DIR__, "figure", "p4_simulation_wage.png"))
println("Saved: p4_simulation_wage.png")


p_e = plot(
    title = "Education Time Over Time",
    xlabel = L"Period\ (after\ burn-in)",
    ylabel = L"Education\ e_t",
    legend = :bottomright,
    grid = true
)

for i in 1:n_paths
    periods = 1:100
    e_plot = all_e[i][start_idx:end_idx]
    plot!(p_e, periods, e_plot, 
          label = "Path $i", 
          color = colors[i], 
          lw = 1.5, 
          alpha = 0.8)
end

savefig(p_e, joinpath(@__DIR__, "figure", "p4_simulation_education.png"))
println("Saved: p4_simulation_education.png")


p_c = plot(
    title = "Consumption Over Time",
    xlabel = L"Period\ (after\ burn-in)",
    ylabel = L"Consumption\ c_t",
    legend = :topright,
    grid = true
)

for i in 1:n_paths
    periods = 1:100
    c_plot = all_c[i][start_idx:end_idx]
    plot!(p_c, periods, c_plot, 
          label = "Path $i", 
          color = colors[i], 
          lw = 1.5, 
          alpha = 0.8)
end

savefig(p_c, joinpath(@__DIR__, "figure", "p4_simulation_consumption.png"))
println("Saved: p4_simulation_consumption.png")

# If we take a look at the plots generated for the path 3, we can clearly see that the output aligns with our expectations. There are sharp decreases in education effort whenever there is a spike in the wage shocks. This behavior is consistent with the model's prediction that higher wages increase the opportunity cost of education, leading individuals to allocate more time to work and less to education.

n_periods_analysis = n_periods - burn_in 


all_y = Float64[]       
all_e_corr = Float64[]  
all_w_corr = Float64[]  
    
for i in 1:n_paths
    for t in (burn_in + 1):n_periods
        h = all_h[i][t]
        w = all_w[i][t]
        e = all_e[i][t]
        y = w * f_h(h, params) * (1 - e)
        push!(all_y, y)
        push!(all_e_corr, e)
        push!(all_w_corr, w)
    end
end

n_obs = length(all_y)
    
corr_y_e = cor(all_y, all_e_corr)
corr_w_e = cor(all_w_corr, all_e_corr)
    
    println("\nCorrelation between labor earnings and education effort:")
println("\ncor(y, e) = $(round(corr_y_e, digits=4))")
    

println("\nCorrelation between wage shock and education effort:")
println("\ncor(w, e) = $(round(corr_w_e, digits=4))")

#The correlation between labor earnings and education effort is negative because when people earn more, the time they spend studying becomes more costly in terms of forgone income. As a result, they tend to shift effort away from education and toward work.
#Similarly, the correlation between wage shocks and education effort is negative because a higher-than-expected wage makes working immediately more attractive. This raises the opportunity cost of education, so individuals are less likely to put effort into their studies.

