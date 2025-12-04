using Random, Distributions, Statistics, Optim, DataFrames, PrettyTables, Plots

function get_seed(inp)::Int64
    h = hash(inp)
    return Int64(mod(h, typemax(Int64)))
end

corr(x,y) = Statistics.cor(x,y)
autocorr1(y::Vector{Float64})::Float64 = corr(y[1:end-1], y[2:end])
kurtΔ(y::Vector{Float64})::Float64 = kurtosis(diff(y))

function eps(p::Float64, σL::Float64, σH::Float64)::Float64
    if rand(Bernoulli(0.8)) == 1
        return rand(Normal(0, σL))
    else
        return rand(Normal(0, σH))
    end
end

function log_income(θ::Vector{Float64}, y::Float64, σL::Float64, σH::Float64)::Float64
    ρ, p = θ
    return ρ * y + eps(p, σL, σH)
end

function simulate_model(θ::Vector{Float64}, T::Int64, σL::Float64, σH::Float64)::Vector{Float64}
    lny = [0.0]
    T += 99 # + 100 burn-in; -1 for first element
    @inbounds for _ in 1:T
        push!(lny, log_income(θ, lny[end], σL, σH))
    end
    return lny[101:end] # remove burn-in
end

# There's a bug in Optim.jl where NelderMead does not respect the box constraints using Fminbox.
# The function works around that by mapping unconstrained θ to the given range.
# Optim.minimizer results should be wrapped with this function tp get parameter estaimtes that respect the bounds.
function θ_bound_logistic(θ::Vector{Float64}, θL::Vector{Float64}=[0.5, 0.5], θH::Vector{Float64}=[0.99, 0.95])::Vector{Float64}
    @inbounds for i in 1:length(θ)
        θ[i] = θL[i] + (θH[i] - θL[i]) / (1 + exp(-θ[i]))
    end
    return θ
end


function smm_objective(θ::Vector{Float64}, obs::Vector{Float64}, σL::Float64, σH::Float64, S::Int64, seed::Int64=0)::Float64
    θ = θ_bound_logistic(θ)
    T = length(obs)
    obs_moments = [
        std(obs),
        autocorr1(obs),
        kurtΔ(obs)
    ]
    
    sim_moments = [Float64[], Float64[], Float64[]]
    
    Random.seed!(get_seed(θ) + seed)
    @inbounds for _ in 1:S
        sim_data = simulate_model(θ, T, σL, σH)
        push!(sim_moments[1], std(sim_data))
        push!(sim_moments[2], autocorr1(sim_data))
        push!(sim_moments[3], kurtΔ(sim_data))
    end
    sim_moments = [mean(m) for m in sim_moments]
    
    return sum((obs_moments .- sim_moments).^2)
end


# ============== Solutions ==============
Random.seed!(2024)

obs = simulate_model([0.9, 0.8], 500, 0.1, 0.3)

θ0 = [0.85, 0.7]
θL = [0.5, 0.5]
θH = [0.99, 0.95]

obj = optimize(
    θ -> smm_objective(θ, obs, 0.1, 0.3, 100),
    θ0,
    NelderMead()
)
θ̂ = θ_bound_logistic(Optim.minimizer(obj))

# Plot Data
sim_data = simulate_model(θ̂, 500, 0.1, 0.3)
p = Plots.plot(1:200, obs[1:200], label="Observed Data", lw=2)
Plots.plot!(1:200, sim_data[1:200], label="Simulated Data", lw=2, title="Observed vs Simulated Data", xlabel="Time", ylabel="Log Income")
out = joinpath(@__DIR__, "figure", "p2_timeseries.png")
savefig(p, out)


Δobs = diff(obs)
Δsim = diff(sim_data)
hist = Plots.histogram(
    [Δobs, Δsim],
    bins = -1.0:0.1:1.0,
    normalize = :pdf,
    label = ["Observed Data" "Simulated Data"],
    title = "Histogram of Income Changes",
    xlabel = "Change in Log Income",
    ylabel = "Density",
    alpha = 0.7,
)
out = joinpath(@__DIR__, "figure", "p2_histogram.png")
savefig(hist, out)

# Monte-Carlo Sim [Very Long Runtime]
M = 100
θ_est50 = zeros(M, 2)
θ_est200 = zeros(M, 2)
@inbounds for i in 1:M
    obs_mc = simulate_model([0.9, 0.8], 500, 0.1, 0.3)
    θ0_mc = [0.85, 0.7]
    obj_mc50 = optimize(
        θ -> smm_objective(θ, obs_mc, 0.1, 0.3, 50, i),
        θ0_mc,
        NelderMead()
    )
    θ_est50[i, :] = θ_bound_logistic(Optim.minimizer(obj_mc50))

    obj_mc200 = optimize(
        θ -> smm_objective(θ, obs_mc, 0.1, 0.3, 200, i),
        θ0_mc,
        NelderMead()
    )
    θ_est200[i, :] = θ_bound_logistic(Optim.minimizer(obj_mc200))

    println("Completed MC iteration $i")
end

std_ρ̂50 = std(θ_est50[:, 1])
std_p̂50 = std(θ_est50[:, 2])
bias_ρ̂50 = mean(θ_est50[:, 1]) - 0.9
bias_p̂50 = mean(θ_est50[:, 2]) - 0.8
df = DataFrame(
    Parameter = ["ρ", "p"],
    Estimate = θ̂,
    Std_Dev = [std_ρ̂, std_p̂],
    Bias = [bias_ρ̂, bias_p̂]
)
pretty_table(df)

s = Plots.scatter(
    θ_est50[:, 1],
    θ_est50[:, 2],
    xlabel = "ρ̂",
    ylabel = "p̂",
    color = :blue,
    ylims=(0.75, 1.0),
    xlims=(0.75, 1.0),
    title = "Monte Carlo Estimates of Parameters",
    label = "S = 50",
    legend = :topright
)

scatter!(
    θ_est200[:, 1],
    θ_est200[:, 2],
    color = :orange,
    label = "S = 200"
)

scatter!(
    [0.9],
    [0.8],
    markershape = :star5,
    markersize = 8,
    color = :red,
    label = "True Value"
)
out = joinpath(@__DIR__, "figure", "p2_monte_carlo.png")
savefig(s, out)