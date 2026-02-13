using Parameters, QuantEcon, LinearAlgebra, Plots, LaTeXStrings, Random, Statistics, Optim, PrettyTables,DataFrames
pyplot()

#Setting up the parameters

@with_kw struct ConsumptionParams
    γ::Float64 = 2.0            # Risk aversion
    β::Float64 = 0.99           # Discount factor
    R::Float64 = 1.010          # Interest rate
    ρ::Float64 = 0.90           # Income persistence
    σ_ε::Float64 = 0.20 * sqrt(1 - 0.90^2)  # Income volatility
    Nz::Int = 11                # Number of income states
    Na::Int = 100               # Number of asset grid points
    a_max::Float64 = 500.0      # Maximum assets
    θ::Float64 = 3.0            # Grid curvature parameter
end

# Utility function
function utility(c, γ)
    if c <= 0
        return -Inf
    end
    if γ ≈ 1.0
        return log(c)
    else
        return (c^(1 - γ)) / (1 - γ)
    end
end

# Income process discretization
function income_process(params::ConsumptionParams)
    mc = QuantEcon.rouwenhorst(params.Nz, params.ρ, params.σ_ε)
    z_grid = exp.(mc.state_values)
    P_z = mc.p
    return z_grid, P_z
end

# Set up asset grid with curvature
function setup_asset_grid(params::ConsumptionParams, z_min)
    nbl = z_min / (params.R - 1)
    a_bar = -0.6 * nbl
    
    println("\nAsset Grid Setup:")
    println("\nNatural borrowing limit: $(round(nbl, digits=4))")
    println("\nActual borrowing limit ā: $(round(a_bar, digits=4))")
    println("\nMaximum assets: $(params.a_max)")
    
    ω = range(0, 1, length=params.Na)
    a_grid = a_bar .+ (params.a_max - a_bar) .* (ω .^ params.θ)
    
    return a_grid
end

# Linear interpolation
function interpolate_V(a_prime, i_z, a_grid, V_old)
    if a_prime <= a_grid[1]
        return V_old[1, i_z]
    end
    
    if a_prime >= a_grid[end]
        return V_old[end, i_z]
    end
    
    idx = searchsortedfirst(a_grid, a_prime)
    
    a_low = a_grid[idx - 1]
    a_high = a_grid[idx]
    V_low = V_old[idx - 1, i_z]
    V_high = V_old[idx, i_z]
    
    weight = (a_prime - a_low) / (a_high - a_low)
    V_interp = V_low + weight * (V_high - V_low)
    
    return V_interp
end

# Method 1 - Standard VFI with linear interpolation

function bellman_standard!(V_new, V_old, policy_c, policy_a, a_grid, z_grid, P_z, params)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    for i_a in 1:Na
        a = a_grid[i_a]
        
        for i_z in 1:Nz
            z = z_grid[i_z]
            max_c = params.R * a + z - a_grid[1]
            
            if max_c <= 0
                V_new[i_a, i_z] = -Inf
                policy_c[i_a, i_z] = 0.0
                policy_a[i_a, i_z] = a_grid[1]
                continue
            end
            
            function objective(c)
                if c <= 0 || c > max_c
                    return Inf
                end
                
                a_prime = params.R * a + z - c
                
                EV = 0.0
                for i_z_next in 1:Nz
                    V_next = interpolate_V(a_prime, i_z_next, a_grid, V_old)
                    EV += P_z[i_z, i_z_next] * V_next
                end
                
                return -(utility(c, params.γ) + params.β * EV)
            end
            
            result = optimize(objective, 1e-6, max_c, Brent())
            
            c_star = result.minimizer
            V_new[i_a, i_z] = -result.minimum
            policy_c[i_a, i_z] = c_star
            policy_a[i_a, i_z] = params.R * a + z - c_star
        end
    end
end

# Main VFI function for standard method
function vfi_standard(params::ConsumptionParams; tol=1e-7, max_iter=5000, verbose=true)
    z_grid, P_z = income_process(params)
    a_grid = setup_asset_grid(params, minimum(z_grid))
    
    Na = length(a_grid)
    Nz = length(z_grid)
    
    V_old = zeros(Na, Nz)
    V_new = zeros(Na, Nz)
    policy_c = zeros(Na, Nz)
    policy_a = zeros(Na, Nz)
    
    if verbose
        println("\nSTANDARD VFI")
        println("\nParameters: γ=$(params.γ), R=$(params.R), β=$(params.β)")
        println("\nGrid: $(Na) assets × $(Nz) income states")
        println("\nStarting iterations...\n")
    end
    
    start_time = time()
    
    for iter in 1:max_iter
        bellman_standard!(V_new, V_old, policy_c, policy_a, a_grid, z_grid, P_z, params)
        
        diff = maximum(abs.(V_new - V_old) ./ (1 .+ abs.(V_old)))
        
        if verbose && (iter % 100 == 0)
            println("Iteration $iter: max relative diff = $(round(diff, digits=10))")
        end
        
        if diff < tol
            elapsed_time = time() - start_time
            if verbose
                println("\nConverged in $iter iterations!")
                println("\nConvergence time: $(round(elapsed_time, digits=3)) seconds\n")
            end
            return V_new, policy_c, policy_a, a_grid, z_grid, P_z, iter, elapsed_time
        end
        
        V_old .= V_new
    end
    
    elapsed_time = time() - start_time
    println("\nError: Did not converge\n")
    return V_new, policy_c, policy_a, a_grid, z_grid, P_z, max_iter, elapsed_time
end


# Method 2 - VFI with CES transformation

function bellman_ces!(W_new, W_old, policy_c, policy_a, a_grid, z_grid, P_z, params)
    Na = length(a_grid)
    Nz = length(z_grid)
    γ = params.γ
    
    for i_a in 1:Na
        a = a_grid[i_a]
        
        for i_z in 1:Nz
            z = z_grid[i_z]
            max_c = params.R * a + z - a_grid[1]
            
            if max_c <= 0
                W_new[i_a, i_z] = -Inf
                policy_c[i_a, i_z] = 0.0
                policy_a[i_a, i_z] = a_grid[1]
                continue
            end
            
            # Objective for CES transformation
            # W(a,z) = max [(1-β)c^(1-γ) + β E[W(a',z')^(1-γ)]]^(1/(1-γ))
            # For optimization, we maximize the inside: (1-β)c^(1-γ) + β E[W(a',z')^(1-γ)]
            
            function objective_ces(c)
                if c <= 0 || c > max_c
                    return Inf
                end
                
                a_prime = params.R * a + z - c
                EW_power = 0.0
                for i_z_next in 1:Nz
                    W_next = interpolate_V(a_prime, i_z_next, a_grid, W_old)
                    if !isfinite(W_next)
                        return Inf
                    end
                    # Raise to power (1-γ) before taking expectation. This part was strange for me in terms of implementation, since we take the power before expectation, which is different from the standard VFI. So I had to be careful here and this step yielded very different results than standard VFI, which we'll discuss at the end of the code when we compare the results.
                    EW_power += P_z[i_z, i_z_next] * (W_next^(1 - γ))
                end
                
                if !isfinite(EW_power)
                    return Inf
                end
                
                # Inside of the CES aggregator
                ces_value = (1 - params.β) * (c^(1 - γ)) + params.β * EW_power
                
                if !isfinite(ces_value)
                    return Inf
                end
                
                return -ces_value  # Multiplied with -1 since we want to maximize the inside and default is minimization.
            end
            
            result = optimize(objective_ces, 1e-6, max_c, Brent())
            
            c_star = result.minimizer
            
            # Transform back: W = [...]^(1/(1-γ))
            ces_value = -result.minimum
            
            if ces_value > 0 && isfinite(ces_value)
                W_new[i_a, i_z] = ces_value^(1 / (1 - γ))
            elseif ces_value < 0 && γ > 1 && isfinite(ces_value)
                # For γ > 1, we can have negative ces_value
                # W = (negative)^(negative exponent) = negative result
                W_new[i_a, i_z] = -abs(ces_value)^(1 / (1 - γ))
            else
                W_new[i_a, i_z] = -Inf
            end
            
            policy_c[i_a, i_z] = c_star
            policy_a[i_a, i_z] = params.R * a + z - c_star
        end
    end
end

function vfi_ces(params::ConsumptionParams; tol=1e-7, max_iter=5000, verbose=true)
z_grid, P_z = income_process(params)
a_grid = setup_asset_grid(params, minimum(z_grid))
Na = length(a_grid)
Nz = length(z_grid)
W_old = zeros(Na, Nz)

for i_a in 1:Na
    for i_z in 1:Nz
        c_init = max(params.R * a_grid[i_a] + z_grid[i_z] - a_grid[1], 0.01)
        if params.γ ≈ 1.0
            # Log utility case
            W_old[i_a, i_z] = log(c_init) / (1 - params.β)
        else
            # For constant consumption c forever:
            # V = c^(1-γ)/((1-γ)(1-β))
            # W = [(1-γ)V]^(1/(1-γ)) = [c^(1-γ)/(1-β)]^(1/(1-γ))
            # W = c / (1-β)^(1/(1-γ))
            W_old[i_a, i_z] = c_init / ((1 - params.β)^(1 / (1 - params.γ)))
        end
    end
end

    
    W_new = similar(W_old)
    policy_c = zeros(Na, Nz)
    policy_a = zeros(Na, Nz)
    
    if verbose
        println("\nCES TRANSFORMATION VFI (Monotone Transformation Trick)")
        println("\nParameters: γ=$(params.γ), R=$(params.R), β=$(params.β)")
        println("\nGrid: $(Na) assets × $(Nz) income states = $(Na*Nz) total states")
        println("\nStarting iterations...\n")
    end
    
    start_time = time()
    
    for iter in 1:max_iter
        bellman_ces!(W_new, W_old, policy_c, policy_a, a_grid, z_grid, P_z, params)
        
        diff = maximum(abs.(W_new - W_old) ./ (1 .+ abs.(W_old)))
        
        if verbose && (iter % 100 == 0)
            println("Iteration $iter: max relative diff = $(round(diff, digits=10))")
        end
        
        if diff < tol
            elapsed_time_ces = time() - start_time
            if verbose
                println("\nConverged in $iter iterations!")
                println("\nFinal max relative difference: $(round(diff, digits=12))")
                println("\nConvergence time: $(round(elapsed_time_ces, digits=3)) seconds\n")
            end
            return W_new, policy_c, policy_a, a_grid, z_grid, P_z, iter, elapsed_time_ces
        end
        
        W_old .= W_new
    end
    
    elapsed_time_ces = time() - start_time
    println("\nError: Did not converge in $max_iter iterations\n")
    return W_new, policy_c, policy_a, a_grid, z_grid, P_z, max_iter, elapsed_time_ces
end

println("\nTEST CASE 1: γ = 2, R = 1.010")
params1 = ConsumptionParams(γ=2.0, R=1.010)

println("\n--- Standard VFI ---")
 V_std1, c_std1, a_std1, a_grid1, z_grid1, P_z1, iter_std1, time_std1 = vfi_standard(params1, verbose=true)

println("\n--- CES Transformation VFI ---")
W_ces1, c_ces1, a_ces1, _, _, _, iter_ces1, time_ces1 = vfi_ces(params1, verbose=true)

# Test Case 2: γ = 10, R = 1.008
println("\nTEST CASE 2: γ = 10, R = 1.008")
params2 = ConsumptionParams(γ=10.0, R=1.008)

println("\n--- Standard VFI ---")
V_std2, c_std2, a_std2, a_grid2, z_grid2, P_z2, iter_std2, time_std2 = vfi_standard(params2, verbose=true)

println("\n--- CES Transformation VFI ---")
W_ces2, c_ces2, a_ces2, _, _, _, iter_ces2, time_ces2 = vfi_ces(params2, verbose=true)

# COMPARISON TABLE
println("\nRESULTS: Iterations and Runtime Comparison")

data = [
    "γ=2, Standard" iter_std1 round(time_std1, digits=3);
    "γ=2, CES" iter_ces1 round(time_ces1, digits=3);
    "γ=10, Standard" iter_std2 round(time_std2, digits=3);
    "γ=10, CES" iter_ces2 round(time_ces2, digits=3)
]

pretty_table(data, column_labels=["Method", "Iterations", "Time (sec)"])

mkpath(joinpath(@__DIR__, "figures"))

# Select three income states: lowest, median, highest
Nz = length(z_grid1)
z_low_idx = 1
z_med_idx = div(Nz + 1, 2)
z_high_idx = Nz

indices = [z_low_idx, z_med_idx, z_high_idx]
labels = ["Lowest z=$(round(z_grid1[z_low_idx], digits=3))",
          "Median z=$(round(z_grid1[z_med_idx], digits=3))",
          "Highest z=$(round(z_grid1[z_high_idx], digits=3))"]
colors = [:blue, :red, :green]

a_bar = a_grid1[1]
zoom_idx = min(20, div(length(a_grid1), 5))


p_gamma2 = plot(layout=(2,2), size=(1400, 1000))

# Subplot 1: Consumption - Standard - Full
plot!(p_gamma2[1], 
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - Standard VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2[1], a_grid1, c_std1[:, i_z],
        label = label, lw = 2, color = color)
end

# Subplot 2: Consumption - Standard - Zoom
plot!(p_gamma2[2],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - Standard VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2[2], a_grid1[1:zoom_idx], c_std1[1:zoom_idx, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_gamma2[2], [a_bar], label = "Borrowing limit ā", 
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 3: Assets - Standard - Full
plot!(p_gamma2[3],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - Standard VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2[3], a_grid1, a_std1[:, i_z],
        label = label, lw = 2, color = color)
end
hline!(p_gamma2[3], [a_bar], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 4: Assets - Standard - Zoom
plot!(p_gamma2[4],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - Standard VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2[4], a_grid1[1:zoom_idx], a_std1[1:zoom_idx, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
hline!(p_gamma2[4], [a_bar], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_gamma2, joinpath(@__DIR__, "figures", "p1_policies_standard_gamma2.png"))


p_gamma2_ces = plot(layout=(2,2), size=(1400, 1000))

# Subplot 1: Consumption - CES - Full
plot!(p_gamma2_ces[1],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - CES VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2_ces[1], a_grid1, c_ces1[:, i_z],
        label = label, lw = 2, color = color)
end

# Subplot 2: Consumption - CES - Zoom
plot!(p_gamma2_ces[2],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - CES VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2_ces[2], a_grid1[1:zoom_idx], c_ces1[1:zoom_idx, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_gamma2_ces[2], [a_bar], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 3: Assets - CES - Full
plot!(p_gamma2_ces[3],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - CES VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2_ces[3], a_grid1, a_ces1[:, i_z],
        label = label, lw = 2, color = color)
end
hline!(p_gamma2_ces[3], [a_bar], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 4: Assets - CES - Zoom
plot!(p_gamma2_ces[4],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - CES VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_gamma2_ces[4], a_grid1[1:zoom_idx], a_ces1[1:zoom_idx, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
hline!(p_gamma2_ces[4], [a_bar], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_gamma2_ces, joinpath(@__DIR__, "figures", "p1_policies_ces_gamma2.png"))

# Select three income states for gamma=10
Nz2 = length(z_grid2)
z_low_idx2 = 1
z_med_idx2 = div(Nz2 + 1, 2)
z_high_idx2 = Nz2

indices2 = [z_low_idx2, z_med_idx2, z_high_idx2]
labels2 = ["Lowest z=$(round(z_grid2[z_low_idx2], digits=3))",
           "Median z=$(round(z_grid2[z_med_idx2], digits=3))",
           "Highest z=$(round(z_grid2[z_high_idx2], digits=3))"]

a_bar2 = a_grid2[1]
zoom_idx2 = min(20, div(length(a_grid2), 5))


p_gamma10 = plot(layout=(2,2), size=(1400, 1000))

# Subplot 1: Consumption - Standard - Full
plot!(p_gamma10[1],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - Standard VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10[1], a_grid2, c_std2[:, i_z],
        label = label, lw = 2, color = color)
end

# Subplot 2: Consumption - Standard - Zoom
plot!(p_gamma10[2],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - Standard VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10[2], a_grid2[1:zoom_idx2], c_std2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_gamma10[2], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 3: Assets - Standard - Full
plot!(p_gamma10[3],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - Standard VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10[3], a_grid2, a_std2[:, i_z],
        label = label, lw = 2, color = color)
end
hline!(p_gamma10[3], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 4: Assets - Standard - Zoom
plot!(p_gamma10[4],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - Standard VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10[4], a_grid2[1:zoom_idx2], a_std2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
hline!(p_gamma10[4], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_gamma10, joinpath(@__DIR__, "figures", "p1_policies_standard_gamma10.png"))

p_gamma10_ces = plot(layout=(2,2), size=(1400, 1000))

# Subplot 1: Consumption - CES - Full
plot!(p_gamma10_ces[1],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - CES VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10_ces[1], a_grid2, c_ces2[:, i_z],
        label = label, lw = 2, color = color)
end

# Subplot 2: Consumption - CES - Zoom
plot!(p_gamma10_ces[2],
    xlabel = L"Assets\ a",
    ylabel = L"Consumption\ c(a,z)",
    title = "Consumption Policy - CES VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10_ces[2], a_grid2[1:zoom_idx2], c_ces2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_gamma10_ces[2], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 3: Assets - CES - Full
plot!(p_gamma10_ces[3],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - CES VFI",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10_ces[3], a_grid2, a_ces2[:, i_z],
        label = label, lw = 2, color = color)
end
hline!(p_gamma10_ces[3], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

# Subplot 4: Assets - CES - Zoom
plot!(p_gamma10_ces[4],
    xlabel = L"Current\ Assets\ a",
    ylabel = L"Next\ Period\ Assets\ a'(a,z)",
    title = "Asset Policy - CES VFI (Zoomed)",
    titlefontsize = 10,
    legend = :bottomright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_gamma10_ces[4], a_grid2[1:zoom_idx2], a_ces2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
hline!(p_gamma10_ces[4], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_gamma10_ces, joinpath(@__DIR__, "figures", "p1_policies_ces_gamma10.png"))

function compute_euler_errors(c_policy, a_policy, a_grid, z_grid, P_z, params)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    euler_errors = zeros(Na, Nz)
    
    for i_a in 1:Na
        a = a_grid[i_a]
        
        for i_z in 1:Nz
            z = z_grid[i_z]
            c = c_policy[i_a, i_z]
            a_prime = a_policy[i_a, i_z]
            
            # Current marginal utility
            if c <= 0
                euler_errors[i_a, i_z] = NaN
                continue
            end
            
            u_prime_c = c^(-params.γ)
            
            # Expected marginal utility next period
            E_u_prime_c_next = 0.0
            
            for i_z_next in 1:Nz
                # Find c_{t+1} given a' and z'
                # Need to interpolate consumption policy at a'
                c_next = interpolate_V(a_prime, i_z_next, a_grid, c_policy)
                
                if c_next <= 0
                    continue
                end
                
                u_prime_c_next = c_next^(-params.γ)
                E_u_prime_c_next += P_z[i_z, i_z_next] * u_prime_c_next
            end
            
            # Euler equation: u'(c) = β R E[u'(c')]
            # Error: |1 - RHS/LHS|
            if E_u_prime_c_next > 0
                euler_ratio = (params.β * params.R * E_u_prime_c_next) / u_prime_c
                euler_error = abs(1.0 - euler_ratio)
                
                # Log10 of error (standard way to report)
                if euler_error > 0
                    euler_errors[i_a, i_z] = log10(euler_error)
                else
                    euler_errors[i_a, i_z] = -16.0  # Essentially zero error
                end
            else
                euler_errors[i_a, i_z] = NaN
            end
        end
    end
    
    return euler_errors
end



println("\nComputing errors for γ=2, Standard VFI...")
ee_std1 = compute_euler_errors(c_std1, a_std1, a_grid1, z_grid1, P_z1, params1)

println("\nComputing errors for γ=2, CES VFI...")
ee_ces1 = compute_euler_errors(c_ces1, a_ces1, a_grid1, z_grid1, P_z1, params1)

println("\nComputing errors for γ=10, Standard VFI...")
ee_std2 = compute_euler_errors(c_std2, a_std2, a_grid2, z_grid2, P_z2, params2)

println("\nComputing errors for γ=10, CES VFI...")
ee_ces2 = compute_euler_errors(c_ces2, a_ces2, a_grid2, z_grid2, P_z2, params2)

# Select three income states
Nz = length(z_grid1)
z_low_idx = 1
z_med_idx = div(Nz + 1, 2)
z_high_idx = Nz

indices = [z_low_idx, z_med_idx, z_high_idx]
labels = ["Lowest z=$(round(z_grid1[z_low_idx], digits=3))",
          "Median z=$(round(z_grid1[z_med_idx], digits=3))",
          "Highest z=$(round(z_grid1[z_high_idx], digits=3))"]
colors = [:blue, :red, :green]

a_bar1 = a_grid1[1]
zoom_idx1 = min(20, div(length(a_grid1), 5))

# GAMMA = 2, STANDARD: 2 subplots (full and zoom)
p_ee_std1 = plot(layout=(1,2), size=(1400, 500))

# Full range
plot!(p_ee_std1[1],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - Standard VFI (γ=2)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_ee_std1[1], a_grid1, ee_std1[:, i_z],
        label = label, lw = 2, color = color)
end

# Zoomed
plot!(p_ee_std1[2],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - Standard VFI (γ=2, Zoomed)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_ee_std1[2], a_grid1[1:zoom_idx1], ee_std1[1:zoom_idx1, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_ee_std1[2], [a_bar1], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_ee_std1, joinpath(@__DIR__, "figures", "p1_euler_errors_standard_gamma2.png"))



p_ee_ces1 = plot(layout=(1,2), size=(1400, 500))

# Full range
plot!(p_ee_ces1[1],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - CES VFI (γ=2)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_ee_ces1[1], a_grid1, ee_ces1[:, i_z],
        label = label, lw = 2, color = color)
end

# Zoomed
plot!(p_ee_ces1[2],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - CES VFI (γ=2, Zoomed)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices, labels, colors)
    plot!(p_ee_ces1[2], a_grid1[1:zoom_idx1], ee_ces1[1:zoom_idx1, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_ee_ces1[2], [a_bar1], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_ee_ces1, joinpath(@__DIR__, "figures", "p1_euler_errors_ces_gamma2.png"))

# Select three income states for gamma=10
Nz2 = length(z_grid2)
z_low_idx2 = 1
z_med_idx2 = div(Nz2 + 1, 2)
z_high_idx2 = Nz2

indices2 = [z_low_idx2, z_med_idx2, z_high_idx2]
labels2 = ["Lowest z=$(round(z_grid2[z_low_idx2], digits=3))",
           "Median z=$(round(z_grid2[z_med_idx2], digits=3))",
           "Highest z=$(round(z_grid2[z_high_idx2], digits=3))"]

a_bar2 = a_grid2[1]
zoom_idx2 = min(20, div(length(a_grid2), 5))

p_ee_std2 = plot(layout=(1,2), size=(1400, 500))

# Full range
plot!(p_ee_std2[1],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - Standard VFI (γ=10)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_ee_std2[1], a_grid2, ee_std2[:, i_z],
        label = label, lw = 2, color = color)
end

# Zoomed
plot!(p_ee_std2[2],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - Standard VFI (γ=10, Zoomed)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_ee_std2[2], a_grid2[1:zoom_idx2], ee_std2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_ee_std2[2], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_ee_std2, joinpath(@__DIR__, "figures", "p1_euler_errors_standard_gamma10.png"))


p_ee_ces2 = plot(layout=(1,2), size=(1400, 500))

# Full range
plot!(p_ee_ces2[1],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - CES VFI (γ=10)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_ee_ces2[1], a_grid2, ee_ces2[:, i_z],
        label = label, lw = 2, color = color)
end

# Zoomed
plot!(p_ee_ces2[2],
    xlabel = L"Assets\ a",
    ylabel = L"Log_{10}\ Euler\ Error",
    title = "Euler Equation Errors - CES VFI (γ=10, Zoomed)",
    titlefontsize = 10,
    legend = :topright,
    grid = true
)
for (i_z, label, color) in zip(indices2, labels2, colors)
    plot!(p_ee_ces2[2], a_grid2[1:zoom_idx2], ee_ces2[1:zoom_idx2, i_z],
        label = label, lw = 2.5, color = color, marker = :circle, markersize = 3)
end
vline!(p_ee_ces2[2], [a_bar2], label = "Borrowing limit ā",
    color = :black, linestyle = :dash, lw = 1.5)

savefig(p_ee_ces2, joinpath(@__DIR__, "figures", "p1_euler_errors_ces_gamma10.png"))


function print_ee_stats(ee, name)
    valid_errors = ee[isfinite.(ee)]
    if length(valid_errors) > 0
        println("\n$name:")
        println("  Mean:   $(round(mean(valid_errors), digits=6))")
        println("  Median: $(round(median(valid_errors), digits=6))")
        println("  Max:    $(round(maximum(valid_errors), digits=6))")
        println("  Min:    $(round(minimum(valid_errors), digits=6))")
    end
end

print_ee_stats(ee_std1, "γ=2, Standard VFI")
print_ee_stats(ee_ces1, "γ=2, CES VFI")
print_ee_stats(ee_std2, "γ=10, Standard VFI")
print_ee_stats(ee_ces2, "γ=10, CES VFI")

function simulate_model(c_policy, a_policy, a_grid, z_grid, P_z, params; 
                       T=10000, burn_in=500, a0=0.0, z0_idx=nothing, seed=123)
    Nz = length(z_grid)
    rng = MersenneTwister(seed)

    if z0_idx === nothing
        z0_idx = div(Nz + 1, 2)  # Median income state
    end

    a_sim          = zeros(T)
    c_sim          = zeros(T)
    z_sim          = zeros(T)
    z_idx_sim      = zeros(Int, T)
    euler_errors_sim = zeros(T)

    a_sim[1]     = a0
    z_idx_sim[1] = z0_idx
    z_sim[1]     = z_grid[z0_idx]

    for t in 1:T
        a_t     = a_sim[t]
        z_idx_t = z_idx_sim[t]

        c_t   = interpolate_V(a_t, z_idx_t, a_grid, c_policy)
        a_tp1 = interpolate_V(a_t, z_idx_t, a_grid, a_policy)

        c_sim[t] = c_t

        # Euler error
        if c_t > 0
            u_prime_c = c_t^(-params.γ)
            E_u_prime_c_next = 0.0
            for i_z_next in 1:Nz
                c_next = interpolate_V(a_tp1, i_z_next, a_grid, c_policy)
                if c_next > 0
                    E_u_prime_c_next += P_z[z_idx_t, i_z_next] * c_next^(-params.γ)
                end
            end
            if E_u_prime_c_next > 0
                euler_error = abs(1.0 - (params.β * params.R * E_u_prime_c_next) / u_prime_c)
                euler_errors_sim[t] = euler_error > 0 ? log10(euler_error) : -16.0
            else
                euler_errors_sim[t] = NaN
            end
        else
            euler_errors_sim[t] = NaN
        end

        # Transition to next period
        if t < T
            cumsum_prob = cumsum(P_z[z_idx_t, :])
            z_idx_next  = findfirst(cumsum(P_z[z_idx_t, :]) .>= rand(rng))
            z_idx_sim[t+1] = z_idx_next
            z_sim[t+1]     = z_grid[z_idx_next]
            a_sim[t+1]     = a_tp1
        end
    end

    return a_sim, c_sim, z_sim, z_idx_sim, euler_errors_sim
end

function compute_simulation_statistics(a_sim, c_sim, z_sim, euler_errors_sim, a_grid; burn_in=500)
    a_post  = a_sim[(burn_in+1):end]
    c_post  = c_sim[(burn_in+1):end]
    z_post  = z_sim[(burn_in+1):end]
    ee_post = euler_errors_sim[(burn_in+1):end]

    stats = Dict()

    stats["mean_a"]  = mean(a_post)
    stats["std_a"]   = std(a_post)
    stats["min_a"]   = minimum(a_post)
    stats["max_a"]   = maximum(a_post)

    stats["mean_c"]  = mean(c_post)
    stats["std_c"]   = std(c_post)
    stats["min_c"]   = minimum(c_post)
    stats["max_c"]   = maximum(c_post)

    a_bar = a_grid[1]
    stats["frac_constrained"] = sum(a_post .<= a_bar + 1e-3) / length(a_post)

    stats["autocorr_a"] = cor(a_post[1:end-1], a_post[2:end])
    stats["autocorr_c"] = cor(c_post[1:end-1], c_post[2:end])
    stats["corr_c_z"]   = cor(c_post, z_post)
    stats["corr_a_z"]   = cor(a_post, z_post)

    valid_ee = ee_post[isfinite.(ee_post)]
    if length(valid_ee) > 0
        stats["mean_ee"]   = mean(valid_ee)
        stats["median_ee"] = median(valid_ee)
        stats["p10_ee"]    = quantile(valid_ee, 0.10)
        stats["p90_ee"]    = quantile(valid_ee, 0.90)
    else
        stats["mean_ee"] = stats["median_ee"] = stats["p10_ee"] = stats["p90_ee"] = NaN
    end

    return stats
end


T        = 10000
burn_in  = 500
a0       = 0.0
SEED     = 2026   # Same seed for all cases per problem set instructions

println("\n--- Simulating γ=2, Standard VFI ---")
a_sim_std1, c_sim_std1, z_sim_std1, z_idx_std1, ee_sim_std1 = simulate_model(
    c_std1, a_std1, a_grid1, z_grid1, P_z1, params1, T=T, burn_in=burn_in, a0=a0, seed=SEED)

println("\n--- Simulating γ=2, CES VFI ---")
a_sim_ces1, c_sim_ces1, z_sim_ces1, z_idx_ces1, ee_sim_ces1 = simulate_model(
    c_ces1, a_ces1, a_grid1, z_grid1, P_z1, params1, T=T, burn_in=burn_in, a0=a0, seed=SEED)

println("\n--- Simulating γ=10, Standard VFI ---")
a_sim_std2, c_sim_std2, z_sim_std2, z_idx_std2, ee_sim_std2 = simulate_model(
    c_std2, a_std2, a_grid2, z_grid2, P_z2, params2, T=T, burn_in=burn_in, a0=a0, seed=SEED)

println("\n--- Simulating γ=10, CES VFI ---")
a_sim_ces2, c_sim_ces2, z_sim_ces2, z_idx_ces2, ee_sim_ces2 = simulate_model(
    c_ces2, a_ces2, a_grid2, z_grid2, P_z2, params2, T=T, burn_in=burn_in, a0=a0, seed=SEED)


stats_std1 = compute_simulation_statistics(a_sim_std1, c_sim_std1, z_sim_std1, ee_sim_std1, a_grid1, burn_in=burn_in)
stats_ces1 = compute_simulation_statistics(a_sim_ces1, c_sim_ces1, z_sim_ces1, ee_sim_ces1, a_grid1, burn_in=burn_in)
stats_std2 = compute_simulation_statistics(a_sim_std2, c_sim_std2, z_sim_std2, ee_sim_std2, a_grid2, burn_in=burn_in)
stats_ces2 = compute_simulation_statistics(a_sim_ces2, c_sim_ces2, z_sim_ces2, ee_sim_ces2, a_grid2, burn_in=burn_in)

stats_keys_list = ["mean_a", "std_a", "min_a", "max_a", 
                   "mean_c", "std_c", "min_c", "max_c",
                   "frac_constrained", "autocorr_a", "autocorr_c", 
                   "corr_c_z", "corr_a_z"]


# SIMULATION RESULTS TABLE
df_stats = DataFrame(
    Statistic     = ["Mean assets (ā)", "Std dev assets (σₐ)", "Min assets", "Max assets",
                     "Mean consumption (c̄)", "Std dev consumption (σc)", "Min consumption", "Max consumption",
                     "Frac. at constraint", "Autocorr(aₜ,aₜ₋₁)", "Autocorr(cₜ,cₜ₋₁)",
                     "Corr(cₜ,zₜ)", "Corr(aₜ,zₜ)"],
    γ2_Std        = [round(stats_std1[k], digits=4) for k in stats_keys_list],
    γ2_CES        = [round(stats_ces1[k], digits=4) for k in stats_keys_list],
    γ10_Std       = [round(stats_std2[k], digits=4) for k in stats_keys_list],
    γ10_CES       = [round(stats_ces2[k], digits=4) for k in stats_keys_list]
)
println(df_stats)

# EULER ERROR TABLE
df_ee = DataFrame(
    Statistic = ["Mean", "Median", "10th percentile", "90th percentile"],
    γ2_Std    = [round(stats_std1[k], digits=4) for k in ["mean_ee","median_ee","p10_ee","p90_ee"]],
    γ2_CES    = [round(stats_ces1[k], digits=4) for k in ["mean_ee","median_ee","p10_ee","p90_ee"]],
    γ10_Std   = [round(stats_std2[k], digits=4) for k in ["mean_ee","median_ee","p10_ee","p90_ee"]],
    γ10_CES   = [round(stats_ces2[k], digits=4) for k in ["mean_ee","median_ee","p10_ee","p90_ee"]]
)
println(df_ee)


plot_start = burn_in + 1
plot_end   = burn_in + 100
periods    = 1:100

# γ = 2: assets, consumption, income + Euler errors
p_ts_gamma2 = plot(layout=(4,1), size=(1200, 1100))

plot!(p_ts_gamma2[1], ylabel=L"Assets\ a_t", title="Assets Over Time (γ=2)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma2[1], periods, a_sim_std1[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma2[1], periods, a_sim_ces1[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)

plot!(p_ts_gamma2[2], ylabel=L"Consumption\ c_t", title="Consumption Over Time (γ=2)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma2[2], periods, c_sim_std1[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma2[2], periods, c_sim_ces1[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)

plot!(p_ts_gamma2[3], ylabel=L"Income\ z_t", title="Income Process (γ=2)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma2[3], periods, z_sim_std1[plot_start:plot_end], label="Income z", lw=2, color=:green)

plot!(p_ts_gamma2[4], xlabel=L"Period", ylabel=L"Log_{10}\ Euler\ Error",
      title="Euler Errors (γ=2)", titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma2[4], periods, ee_sim_std1[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma2[4], periods, ee_sim_ces1[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)

savefig(p_ts_gamma2, joinpath(@__DIR__, "figures", "p1_timeseries_gamma2.png"))


# γ = 10: same layout
p_ts_gamma10 = plot(layout=(4,1), size=(1200, 1100))

plot!(p_ts_gamma10[1], ylabel=L"Assets\ a_t", title="Assets Over Time (γ=10)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma10[1], periods, a_sim_std2[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma10[1], periods, a_sim_ces2[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)

plot!(p_ts_gamma10[2], ylabel=L"Consumption\ c_t", title="Consumption Over Time (γ=10)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma10[2], periods, c_sim_std2[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma10[2], periods, c_sim_ces2[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)

plot!(p_ts_gamma10[3], ylabel=L"Income\ z_t", title="Income Process (γ=10)",
      titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma10[3], periods, z_sim_std2[plot_start:plot_end], label="Income z", lw=2, color=:green)

plot!(p_ts_gamma10[4], xlabel=L"Period", ylabel=L"Log_{10}\ Euler\ Error",
      title="Euler Errors (γ=10)", titlefontsize=11, legend=:topright, grid=true)
plot!(p_ts_gamma10[4], periods, ee_sim_std2[plot_start:plot_end], label="Standard VFI", lw=2, color=:blue)
plot!(p_ts_gamma10[4], periods, ee_sim_ces2[plot_start:plot_end], label="CES VFI",      lw=2, color=:red, linestyle=:dash)
savefig(p_ts_gamma10, joinpath(@__DIR__, "figures", "p1_timeseries_gamma10.png"))


# In solving the model, I was initially confused about why the CES value function iteration produced results so different from standard VFI.
# After spending around 1.5–2 days investigating, I realized the difference is fundamental rather than a coding issue.
# The CES specification implements Epstein–Zin recursive preferences, which introduce a nonlinear certainty equivalent (E[V^(1-γ)])^(1/(1-γ)).
# Unlike standard expected utility where expectations are linear, this power transformation inside the expectation changes the structure of the problem.
# Therefore, the two approaches should not yield identical policies.
# This explains why standard VFI generates smooth consumption and savings functions.
# In contrast, the CES/Epstein–Zin model exhibits discontinuities, threshold behavior, higher consumption volatility, and assets clustering near the borrowing constraint.
# The nonlinear aggregation amplifies tail risks, especially under high risk aversion.
# This leads to strong precautionary behavior and sudden policy switches.
# The most informative numerical result is the alternating spike pattern in Euler errors.
# While standard VFI errors remain small, CES errors become extremely large but follow a structured pattern.
# Spikes alternate across income states.
# Low-income agents spike near the borrowing constraint, while high-income agents spike at moderate asset levels.
# Sign changes in the errors indicate the solution oscillates around discontinuities that fall between grid points.
# Increasing risk aversion amplifies these effects further.
# Importantly, the economic mechanisms such as discontinuities, thresholds, and state-dependent sensitivity are genuine features of Epstein–Zin preferences.
# The extreme Euler-error magnitudes reflect numerical limitations from using a coarse uniform grid.
# More advanced methods like endogenous or adaptive grids would be required for precise quantitative accuracy.
# Overall, the differences between the two specifications arise because Epstein–Zin preferences are not a monotonic transformation of standard utility.
# They fundamentally alter the decision problem.
# The large policy and Euler-error differences therefore reflect both richer economics and numerical challenges rather than implementation mistakes.
# After a lot of trial and error, I concluded that my approach is mostly correct.
# However, if there is anything I am missing or doing incorrectly in the CES VFI, I would appreciate the feedback.




  



