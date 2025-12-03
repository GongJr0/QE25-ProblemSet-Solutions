using Random, Distributions, Statistics, Optim, DataFrames, PrettyTables, Plots, PyPlot

corr(x,y) = Statistics.cor(x,y)
autocorr1(y::Vector{Float64})::Float64 = corr(y[1:end-1], y[2:end])
kurtΔ(y::Vector{Float64})::Float64 = kurtosis(diff(y))

function pseudo_unique_seed(inp)
    h = hash(inp)
    return Int(mod(h, typemax(Int64)))
end

function eps(p::Float64=0.8, σL::Float64=0.1, σH::Float64=0.3, rand_seed::Int64=-1)::Float64
    if rand_seed < 0
        seed = pseudo_unique_seed((p, σL, σH))
    else
        seed = rand_seed
    end

    if rand() <= p
        return rand(Normal(0, σL))
    else
        return rand(Normal(0, σH))
    end
end

function log_inc(ρ::Float64, logy::Float64, eps_gen::Function)::Float64
    ε = eps_gen()
    return ρ*logy + ε
end

function simulate_model(θ::Vector{Float64}, T::Int64, σL::Float64, σH::Float64, rand_seed::Int64=-1)::Vector{Float64}
    T += 99  # +100 for burn in, -1 for y₀
    y_ls = [0.0]
    ε() = eps(θ[2], σL, σH, rand_seed)
    for _ in 1:T
        y_next = log_inc(θ[1], y_ls[end], ε)
        push!(y_ls, y_next)
    end
    return y_ls[101:end]
end

function smm_objective(θ::Vector{Float64}, obs, σL::Float64, σH::Float64, S::Int64)::Float64
    Random.seed!(pseudo_unique_seed(θ))

    sim_std = []
    sim_autocorr = []
    sim_kurtΔ = []
    
    T = length(obs)
    for _ in 1:S
        seed = Int(round(rand()*10000))
        y = simulate_model(θ, T, σL, σH, seed)
        
        push!(sim_std, std(y))
        push!(sim_autocorr, autocorr1(y))
        push!(sim_kurtΔ, kurtΔ(y))
    end
    moments_sim = [
        mean(sim_std),
        mean(sim_autocorr),
        mean(sim_kurtΔ)
    ]
    moments_obs = [
        std(obs),
        autocorr1(obs),
        kurtΔ(obs)
    ]
    return sum((moments_obs .- moments_sim).^2)

end

function estimate_smm(
    θ0::Vector{Float64},
    obs::Vector{Float64},
    σL::Float64,
    σH::Float64,
    S::Int64
)
    obj(θ) = smm_objective(θ, obs, σL, σH, S)
    result = optimize(obj, [0.5, 0.5], [0.99, 0.95], θ0, Fminbox())
    θ_est = Optim.minimizer(result)
    return (θ_est[1], θ_est[2])
end

# ============= Solutions =============

T=500
θ = [0.9, 0.8]

y_obs = simulate_model(θ, T, 0.1, 0.3, 2024)

θ̂ = estimate_smm([0.85, 0.7], y_obs, 0.1, 0.3, 50)
