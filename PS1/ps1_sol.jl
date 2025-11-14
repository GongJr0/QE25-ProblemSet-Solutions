using Random, Distributions, StatsPlots, StatsBase


# ============== Problem 1 ==============

function get_pois_z(n, nrep=1000, λ=1)
    X = rand(Poisson(λ), n, nrep)     # n×nrep
    x̄ = vec(mean(X, dims = 1))
    μ = λ
    σ = sqrt(λ)
    z = (x̄ .- μ) ./ (σ * sqrt(n))
    return z
end

function plot_pois(z, n)
    histogram(
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


ns      = [5, 25, 100, 1000]   # sample sizes
nrep    = 1000                 # number of Monte Carlo replications
z_range = -4:0.01:4            # range for the normal density

plots = Any[]

for n in ns
    z = get_pois_z(n, nrep, λ)

    p = plot_pois(z, n)
    push!(plots, p)
end

plot(plots..., layout = (2, 2), size = (800, 600))