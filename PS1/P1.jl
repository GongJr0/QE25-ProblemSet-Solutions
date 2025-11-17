using Random, Distributions, Statistics, Plots
Random.seed!(0)

# ============== Problem 1 ==================
function get_pois_z(n, nrep=1000, λ=1)
    X = rand(Poisson(λ), n, nrep)     # n × nrep
    x̄ = vec(mean(X, dims = 1))
    μ = λ
    σ = sqrt(λ)
    z = (x̄ .- μ) .* sqrt(n) ./ σ
    return z
end

function plot_pois(z, n)
    p = histogram(
        z;
        bins = -4:0.5:4,
        normalize = :pdf,
        legend = false,
        xlabel = "z",
        ylabel = "Frequency",
        title = "N = $n",
        alpha = 1,
    )
    xlims!(-4, 4)
    ylims!(0, 1)
    z_range = -4:0.01:4
    plot!(z_range, pdf.(Normal(0, 1), z_range), lw = 2, color=:red)
    
    return p
end

λ = 1.0
μ = λ
σ = sqrt(λ)


ns = [5, 25, 100, 1000]
nrep = 1000
z_range = -4:0.01:4

plots = Any[]

for n in ns
    z = get_pois_z(n, nrep, λ)

    p = plot_pois(z, n)
    push!(plots, p)
end

plot(plots..., layout = (2, 2))

out = joinpath(@__DIR__, "figure", "problem1.png")
savefig(out)