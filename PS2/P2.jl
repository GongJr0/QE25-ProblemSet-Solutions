using Random, Distributions, Optim, DataFrames, PrettyTables, Plots
Random.seed!(2024)

# ============== Function Definitions ==============
function piecewiese_eps(p::Float64=0.8, σ1::Float64=0.1, σ2::Float64=0.3)::Float64
   x = rand(Bernoulli(p))
    if x == 1
        return rand(Normal(0, σ1))
    else
        return rand(Normal(0, σ2))
    end
end

function log_inc(ρ::Float64, logy::Float64, eps_gen::Function)::Float64
    ε = eps_gen()
    return ρ*logy + ε
end

function simulate_model(θ::Tuple{Float64, Float64}, T::Int64, σL::Float64, σH::Float64)
    ρ, p = θ
    logy = [0.0]
    T -= 1
    eps_gen() = piecewiese_eps(p, σL, σH)
    for _ in 1:T
        y_new = log_inc(ρ, logy[end], eps_gen)
        push!(logy, y_new)
    end
    return logy[101:end]
end

function autocorr1(y::Vector{Float64})::Float64
    y_lag1 = y[1:end-1]
    y_t = y[2:end]
    return cor(y_t, y_lag1)
end

function pseudo_unique_seed(θ::Tuple{Float64,Float64})::Int64
    h = hash(θ)
    return Int(mod(h, typemax(Int64)))
end

function smm_objective(θ::Tuple{Float64, Float64}, obs::Vector{Float64}, σL::Float64, σH::Float64, S::Int64=100)
    Random.seed!(pseudo_unique_seed(θ))  # deterministic and probabilistically unique seed per {ρ, p} pair.

    T = length(obs) + 100  # Add burn-in period
    sims = [simulate_model(θ, T, σL, σH) for _ in 1:S]
    
    sim_std = mean(std.(sims))
    sim_autocorr = mean(autocorr1.(sims))
    sim_kurt = mean(kurtosis.(diff.(sims)))
    
    moments_sim = [sim_std, sim_autocorr, sim_kurt]
    moments_obs = [std(obs), autocorr1(obs), kurtosis(diff(obs))]
    
    return sum((moments_obs .- moments_sim).^2)
end

function estimate_smm(
    θ0::Tuple{Float64, Float64},
    obs::Vector{Float64},
    σL::Float64,
    σH::Float64;
    S::Int64 = 100
)   
    obj(θ_vec) = smm_objective((θ_vec[1], θ_vec[2]), obs, σL, σH, S)

    lower = [0.5, 0.5]
    upper = [0.99, 0.95]
    θ0_vec = [θ0[1], θ0[2]]

    res = optimize(obj, lower, upper, θ0_vec)
    θ̂ = Optim.minimizer(res)
    return (θ̂[1], θ̂[2])
end

# ============== Simulation ==============
T = 499
ρ = 0.9
logy = [0.0]

for _ in 1:T
    ln_y = log_inc(ρ, logy[end], piecewiese_eps)
    push!(logy, ln_y)
end
logy = logy[101:end]
std_logy = std(logy)

logy_lag1 = logy[1:end-1]
logy_t = logy[2:end]

autocorr = cor(logy_t, logy_lag1)
kurt_Δy = kurtosis(diff(logy_t))

estimate_smm((0.85, 0.7), logy, 0.1, 0.3; S=100)