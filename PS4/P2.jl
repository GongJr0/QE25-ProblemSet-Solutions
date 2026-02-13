using Parameters, QuantEcon, LinearAlgebra, Plots, LaTeXStrings, Random, Statistics, Optim, PrettyTables, DataFrames
pyplot()

# For my comments about the task in general please check the end.

# Copy parameters and utility function from Problem 1
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

# Linear interpolation helper
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

# Standard VFI Bellman operator
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
        println("\nStarting iterations...")
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
                println("Convergence time: $(round(elapsed_time, digits=3)) seconds\n")
            end
            return V_new, policy_c, policy_a, a_grid, z_grid, P_z, iter, elapsed_time
        end
        
        V_old .= V_new
    end
    
    elapsed_time = time() - start_time
    println("\nError: Did not converge\n")
    return V_new, policy_c, policy_a, a_grid, z_grid, P_z, max_iter, elapsed_time
end

# Compute indiviudal MPC'S
function compute_mpc(a, z_idx, delta, c_policy, a_grid, z_grid, params)
    # Current consumption at (a, z)
    c_current = interpolate_V(a, z_idx, a_grid, c_policy)
    
    # Consumption at (a + delta, z) - need to interpolate
    a_plus_delta = a + delta
    c_plus_delta = interpolate_V(a_plus_delta, z_idx, a_grid, c_policy)
    
    # MPC = change in consumption / transfer size
    mpc = (c_plus_delta - c_current) / delta
    
    return mpc
end

# Compute MPC grid for all (a, z) and multiple deltas
function compute_mpc_grid(c_policy, a_grid, z_grid, params, deltas)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    # Store MPC matrices for each delta
    mpc_dict = Dict{Float64, Matrix{Float64}}()
    
    for delta in deltas
        mpc_matrix = zeros(Na, Nz)
        
        for i_a in 1:Na
            for i_z in 1:Nz
                mpc_matrix[i_a, i_z] = compute_mpc(a_grid[i_a], i_z, delta, 
                                                    c_policy, a_grid, z_grid, params)
            end
        end
        
        mpc_dict[delta] = mpc_matrix
    end
    
    return mpc_dict
end



#Compute Stationary distribution using the policy function and transition matrix
function compute_stationary_distribution(a_policy, a_grid, z_grid, P_z; 
                                        tol=1e-10, max_iter=10000, verbose=true)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    # Initialize with uniform distribution
    λ_old = ones(Na, Nz) / (Na * Nz)
    λ_new = zeros(Na, Nz)
    
    if verbose
        println("\nComputing stationary distribution...")
    end
    
    for iter in 1:max_iter
        fill!(λ_new, 0.0)
        
        # For each current state (a, z)
        for i_a in 1:Na
            for i_z in 1:Nz
                mass = λ_old[i_a, i_z]
                
                if mass < 1e-15
                    continue
                end
                
                # Get next period assets
                a_prime = a_policy[i_a, i_z]
                
                # Find position in grid for a_prime, mass splitting
                if a_prime <= a_grid[1]
                    i_a_next = 1
                    weight_low = 1.0
                    weight_high = 0.0
                    i_a_next_high = 1
                elseif a_prime >= a_grid[end]
                    i_a_next = Na
                    weight_low = 1.0
                    weight_high = 0.0
                    i_a_next_high = Na
                else
                    # Find bracketing indices
                    i_a_next = searchsortedfirst(a_grid, a_prime) - 1
                    i_a_next_high = i_a_next + 1
                    
                    # Linear interpolation weights
                    weight_high = (a_prime - a_grid[i_a_next]) / 
                                 (a_grid[i_a_next_high] - a_grid[i_a_next])
                    weight_low = 1.0 - weight_high
                end
                
                # Distribute mass to next period states
                for i_z_next in 1:Nz
                    prob_z = P_z[i_z, i_z_next]
                    
                    # Add mass to lower grid point
                    λ_new[i_a_next, i_z_next] += mass * prob_z * weight_low
                    
                    # Add mass to upper grid point (if different)
                    if i_a_next_high != i_a_next
                        λ_new[i_a_next_high, i_z_next] += mass * prob_z * weight_high
                    end
                end
            end
        end
        
        # Check convergence
        diff = maximum(abs.(λ_new - λ_old))
        
        if verbose && (iter % 500 == 0)
            println("  Iteration $iter: max diff = $(round(diff, digits=12))")
        end
        
        if diff < tol
            if verbose
                println("\nConverged in $iter iterations!\n")
            end
            
            # Normalize (should already sum to 1, but numerical precision)
            λ_new ./= sum(λ_new)
            
            return λ_new
        end
        
        λ_old .= λ_new
    end
    
    println("\nError: Stationary distribution did not converge in $max_iter iterations")
    λ_new ./= sum(λ_new)
    return λ_new
end

#Apply transfer to the distribution: shift mass according to transfer size and interpolate
function apply_transfer_to_distribution(λ_steady, delta, a_grid, z_grid)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    λ_0 = zeros(Na, Nz)
    
    # For each state in steady state distribution
    for i_a in 1:Na
        for i_z in 1:Nz
            mass = λ_steady[i_a, i_z]
            
            if mass < 1e-15
                continue
            end
            
            # After transfer, assets are a + delta
            a_new = a_grid[i_a] + delta
            
            # Find position in grid (mass splitting)
            if a_new <= a_grid[1]
                λ_0[1, i_z] += mass
            elseif a_new >= a_grid[end]
                λ_0[Na, i_z] += mass
            else
                # Linear interpolation
                i_a_low = searchsortedfirst(a_grid, a_new) - 1
                i_a_high = i_a_low + 1
                
                weight_high = (a_new - a_grid[i_a_low]) / 
                             (a_grid[i_a_high] - a_grid[i_a_low])
                weight_low = 1.0 - weight_high
                
                λ_0[i_a_low, i_z] += mass * weight_low
                λ_0[i_a_high, i_z] += mass * weight_high
            end
        end
    end
    
    # Normalize
    λ_0 ./= sum(λ_0)
    
    return λ_0
end

# Evolve distribution forward using policy function and transition matrix
function evolve_distribution_forward(λ_current, a_policy, a_grid, z_grid, P_z)
    Na = length(a_grid)
    Nz = length(z_grid)
    
    λ_next = zeros(Na, Nz)
    
    for i_a in 1:Na
        for i_z in 1:Nz
            mass = λ_current[i_a, i_z]
            
            if mass < 1e-15
                continue
            end
            
            a_prime = a_policy[i_a, i_z]
            
            # Find position in grid
            if a_prime <= a_grid[1]
                i_a_next = 1
                weight_low = 1.0
                weight_high = 0.0
                i_a_next_high = 1
            elseif a_prime >= a_grid[end]
                i_a_next = Na
                weight_low = 1.0
                weight_high = 0.0
                i_a_next_high = Na
            else
                i_a_next = searchsortedfirst(a_grid, a_prime) - 1
                i_a_next_high = i_a_next + 1
                
                weight_high = (a_prime - a_grid[i_a_next]) / 
                             (a_grid[i_a_next_high] - a_grid[i_a_next])
                weight_low = 1.0 - weight_high
            end
            
            # Distribute mass
            for i_z_next in 1:Nz
                prob_z = P_z[i_z, i_z_next]
                λ_next[i_a_next, i_z_next] += mass * prob_z * weight_low
                
                if i_a_next_high != i_a_next
                    λ_next[i_a_next_high, i_z_next] += mass * prob_z * weight_high
                end
            end
        end
    end
    
    λ_next ./= sum(λ_next)
    return λ_next
end

#Compute aggregate consumption given distribution and policy function
function compute_aggregate_consumption(λ, c_policy, a_grid, z_grid)
    C_agg = 0.0
    
    for i_a in 1:length(a_grid)
        for i_z in 1:length(z_grid)
            C_agg += c_policy[i_a, i_z] * λ[i_a, i_z]
        end
    end
    
    return C_agg
end



#Simulate aggregate transfer response: compute stationary distribution, apply transfer, evolve forward, and compute consumption path
function simulate_aggregate_transfer(c_policy, a_policy, a_grid, z_grid, P_z, params;
                                    transfer_fraction=0.05, T_periods=50, verbose=true)
    
    # Compute stationary distribution
    λ_steady = compute_stationary_distribution(a_policy, a_grid, z_grid, P_z, 
                                               verbose=verbose)
    
    # Compute steady state aggregate consumption
    C_steady = compute_aggregate_consumption(λ_steady, c_policy, a_grid, z_grid)
    
    if verbose
        println("Steady state aggregate consumption: $(round(C_steady, digits=6))")
    end
    
    # Determine transfer size
    Δ = transfer_fraction * C_steady
    
    if verbose
        println("Transfer size ($(transfer_fraction*100)% of C*): $(round(Δ, digits=6))\n")
    end
    
    # Apply transfer at t=0
    λ_0 = apply_transfer_to_distribution(λ_steady, Δ, a_grid, z_grid)
    
    # Compute consumption at t=0
    C_0 = compute_aggregate_consumption(λ_0, c_policy, a_grid, z_grid)
    
    # Initialize storage
    C_path = zeros(T_periods + 1)
    C_path[1] = C_0
    
    λ_path = Vector{Matrix{Float64}}(undef, T_periods + 1)
    λ_path[1] = copy(λ_0)
    
    # Evolve distribution forward for t=1,...,T_periods
    if verbose
        println("Simulating impulse response...")
    end
    
    λ_current = copy(λ_0)
    
    for t in 1:T_periods
        # Evolve distribution
        λ_next = evolve_distribution_forward(λ_current, a_policy, a_grid, z_grid, P_z)
        
        # Compute consumption
        C_t = compute_aggregate_consumption(λ_next, c_policy, a_grid, z_grid)
        
        # Store
        λ_path[t+1] = copy(λ_next)
        C_path[t+1] = C_t
        
        # Update for next iteration
        λ_current = λ_next
        
        if verbose && (t % 10 == 0)
            println("  Period $t: C = $(round(C_t, digits=6)), " *
                   "deviation = $(round(100*(C_t - C_steady)/C_steady, digits=4))%")
        end
    end
    
    if verbose
        println("\nImpulse response simulation complete!\n")
    end
    
    return C_path, λ_path, C_steady, Δ
end

# Compute cumulative MPCs for specified horizons
function compute_cumulative_mpcs(C_path, C_steady, delta, horizons)
    cumulative_mpcs = Dict{Int, Float64}()
    
    for H in horizons
        if H == 0
            # Impact MPC
            cumulative_mpcs[H] = (C_path[1] - C_steady) / delta
        else
            # Sum from t=0 to t=H-1 (inclusive)
            cumulative_response = sum(C_path[1:H] .- C_steady)
            cumulative_mpcs[H] = cumulative_response / delta
        end
    end
    
    return cumulative_mpcs
end

# Use γ=2 with Standard VFI (best accuracy from Problem 1)
params = ConsumptionParams(γ=2.0, R=1.010)

# Solve the model
println("\nSOLVING THE MODEL (Standard VFI, γ=2)...")
V, c_policy, a_policy, a_grid, z_grid, P_z, iter, elapsed = vfi_standard(params, verbose=true)

# Transfer sizes to analyze
deltas = [0.01, 0.1, 0.5, 1.0, 2.0]

println("\nComputing MPCs for transfer sizes: $deltas")
mpc_dict = compute_mpc_grid(c_policy, a_grid, z_grid, params, deltas)


# Select three income states
Nz = length(z_grid)
z_low_idx = 1
z_med_idx = div(Nz + 1, 2)
z_high_idx = Nz

indices = [z_low_idx, z_med_idx, z_high_idx]
labels = ["Lowest z=$(round(z_grid[z_low_idx], digits=3))",
          "Median z=$(round(z_grid[z_med_idx], digits=3))",
          "Highest z=$(round(z_grid[z_high_idx], digits=3))"]
colors = [:blue, :red, :green]

# Plot MPCs for each delta
println("\nGenerating MPC plots...")

for delta in deltas
    p = plot(
        xlabel = L"Assets\ a",
        ylabel = L"MPC(a,z;\Delta)",
        title = "Marginal Propensity to Consume (Δ = $delta)",
        titlefontsize = 10,
        legend = :topright,
        grid = true,
        size = (800, 600)
    )
    
    mpc_matrix = mpc_dict[delta]
    
    for (i_z, label, color) in zip(indices, labels, colors)
        plot!(p, a_grid, mpc_matrix[:, i_z],
              label = label, lw = 2, color = color)
    end
    
    # Add horizontal line at MPC = 1 for reference
    hline!(p, [1.0], label = "MPC = 1", color = :black, 
           linestyle = :dash, lw = 1.5)
    
    savefig(p, joinpath(@__DIR__, "figures", "p2_mpc_delta_$(delta).png"))
end

# Summary statistics for MPCs

for delta in deltas
    println("\nTransfer size Δ = $delta:")
    mpc_mat = mpc_dict[delta]
    
    # Stats by income state
    for (idx, name) in zip([z_low_idx, z_med_idx, z_high_idx], 
                           ["Lowest income", "Median income", "Highest income"])
        mpcs = mpc_mat[:, idx]
        println("  $name: mean=$(round(mean(mpcs), digits=4)), " *
               "median=$(round(median(mpcs), digits=4)), " *
               "min=$(round(minimum(mpcs), digits=4)), " *
               "max=$(round(maximum(mpcs), digits=4))")
    end
    
    # Overall stats
    println("  Overall: mean=$(round(mean(mpc_mat), digits=4)), " *
           "median=$(round(median(mpc_mat), digits=4))")
end



# Simulate aggregate transfer (5% of steady state consumption)
C_path, λ_path, C_steady, Δ = simulate_aggregate_transfer(
    c_policy, a_policy, a_grid, z_grid, P_z, params,
    transfer_fraction=0.05, T_periods=50, verbose=true
)

# Compute impulse response as percentage deviation from steady state
impulse_response = 100 .* (C_path .- C_steady) ./ C_steady


p_impulse = plot(
    0:50, impulse_response,
    xlabel = L"Periods\ after\ transfer",
    ylabel = L"Percent\ deviation\ from\ steady\ state\ (\%)",
    title = "Aggregate Consumption Response to 5% Transfer",
    titlefontsize = 10,
    legend = false,
    lw = 2.5,
    color = :blue,
    marker = :circle,
    markersize = 2,
    grid = true,
    size = (900, 600)
)

hline!(p_impulse, [0.0], color = :black, linestyle = :dash, lw = 1.5)

savefig(p_impulse, joinpath(@__DIR__, "figures", "p2_impulse_response.png"))

# Compute cumulative MPCs
horizons = [0, 1, 4, 8, 12, 20]
cumulative_mpcs = compute_cumulative_mpcs(C_path, C_steady, Δ, horizons)


for H in horizons
    mpc_H = cumulative_mpcs[H]
    pct_spent = 100 * mpc_H
    println(@sprintf("    %2d      |     %.4f      |      %.2f%%", H, mpc_H, pct_spent))
end

# Fraction spent by H=12
fraction_spent_12 = cumulative_mpcs[12]
println("\nBy 12 periods after the transfer, approximately " *
       "$(round(fraction_spent_12*100, digits=2))% of the transfer has been spent.")
# Impact MPC
impact_mpc = (C_path[1] - C_steady) / Δ
println("\nIMPACT MPC (period 0): $(round(impact_mpc, digits=4))")
println("This means $(round(impact_mpc*100, digits=2))% of the transfer " *
       "is consumed immediately.")

# I decided to use the standard VFI method since it produced smaller Euler errors.

# The spikes in the policy functions under the CES trick were not satisfactory to me.

# Although the standard VFI could be improved—such as by adding more grid points for assets or implementing more robust methods like HFI or the Endogenous Grid Method—I chose to proceed with it due to time constraints.

# From the plots, my main observation is that MPCs are higher for lower-income states.

# The intuition is straightforward: these households are primarily concerned with survival, so when they receive an additional dollar of income, they tend to spend it immediately.

# In contrast, higher-income households are more likely to save extra income since immediate consumption is less critical for them.

# The MPC values appear somewhat low, as I would normally expect households to spend additional income right away.

# However, I believe this result is related to the model design and the borrowing constraint.

# Since maximum debt is scaled by 0.6, agents are not extremely poor, which dampens the immediate consumption response.

# Finally, the IRF aligns with my expectations for a one-time transfer: consumption rises on impact and then gradually declines back toward the steady state.

# However, because this adjustment occurs over roughly 1200–1300 iterations, the full convergence is not visible in the plot.




