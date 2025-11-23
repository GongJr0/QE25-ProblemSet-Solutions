using Random, Statistics, Distributions, Plots, DataFrames, LaTeXStrings, CSV, LinearAlgebra, IterativeSolvers, PrettyTables
# Problem 1
function clt(n, λ=1, r=1000)
    standardized = Float64[]
    for i in 1:r
        sample = rand(Poisson(λ), n)
        sm = mean(sample)
        z = (sm - λ) / (sqrt(λ) / sqrt(n))
        push!(standardized, z)
    end
    return standardized
end

function plot_clt()
    ns = [5, 25, 100, 1000]
    p = []           
    for n in ns
        data = clt(n)
        h = histogram(data,
            normalize=:pdf,
            bins=20,
            title="N = $n",
            xlabel="z",
            ylabel="Frequency",
            legend=false,
            fillcolor=:dodgerblue,
            linecolor=:black,
            linewidth=0.5,
            titlefontsize=14,
            xlabelfontsize=12,
            ylabelfontsize=12,
            grid=true,
            gridalpha=0.3,
            gridstyle=:solid,
            framestyle=:box)
        
        # Plot normal distribution curve
        plot!(h, -4:0.01:4, pdf.(Normal(0,1), -4:0.01:4),
            color=:red, 
            linewidth=3,
            linestyle=:solid)
        
        xlims!(-4, 4)
        ylims!(0, 1)
        push!(p, h)
    end
    
    plot(p..., layout=(2,2), size=(800, 600), 
         plot_title="", 
         margin=5Plots.mm)
end

plot_clt()
savefig("figures/clt_poisson.png")