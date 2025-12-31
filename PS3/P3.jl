using Plots, LaTeXStrings
pyplot() # Q4 subplots look weird with default backend

# ====== Problem Setup (Params, Structs, Funcs) ======

struct AgentState
    DEAD::Int8
    ALIVE::Int8
end

A = AgentState(-1, 1)
A_idx(state::Int8) = state == A.DEAD ? 1 : 2

# J   = 60 -> Parametrized into solve_model & simulate_lifecycle
γ   = 2.0
γ_b = 1.0
β   = 0.96
r   = 0.04167
ā   = 2
# θ   = 0.5 -> Parametrized into the function ϕ(a)
# ȳ   = 1.0 -> Parametrized into the functions y(j) & c(a_now, a_next, j)

# Consumption inferred from budget constraint
c(a_now::Float64, a_next::Float64, j::Int, ȳ::Float64=1.0) = (1+r)*a_now + y(j, ȳ) - a_next

# P(DEATH | j)
π(j::Int) = j==J ? 1 : min(5e-4 * 1.14^j, 1)

# Period Utility
u(c::Float64) = c<0 ? -Inf : (c^(1-γ))/(1-γ)

# Terminal Utility (CRRA with log case)
# Log case wasn't defined in problem set but without it: (1-γ_b)=0 → ϕ(a)=Undef ∀ a 

ϕ(a::Float64, θ::Float64=0.5) = θ * ( isapprox(γ_b, 1) ? log(a+ā) : ( (a+ā)^(1-γ_b) / (1-γ_b) ) )

# Income Profile
y(j::Int, ȳ::Float64=1.0) = j<= 40 ? (0.8+0.02*j)*ȳ : 0.3*ȳ

function compute_amax(ȳ::Float64=1.0)
    a = 0.0
    for j in 1:J
        a = (1+r)*a + y(j, ȳ)
    end
    return a
end

# ====== Model Solution ======

function solve_model(J::Int=60, ȳ::Float64=1.0, θ::Float64=0.5)
    V = fill(-Inf, J, 500)
    apol = fill(0.0, J, 500)
    cpol = fill(0.0, J, 500)

    amax = compute_amax(ȳ)
    agrid = [amax * ((i-1)/(500-1))^2 for i in 1:500]

    for j in J:-1:1
        πj = π(j)

        for i in 1:500
            a = agrid[i]
            res = (1+r)*a + y(j, ȳ)

            V_best = -Inf
            cbest = 0.0
            anbest = 0.0
            
            for k in 1:500
                an = agrid[k]
                if an > res
                    break
                end

                cj = c(a, an, j, ȳ)
                u_flow = u(cj)
                if u_flow == -Inf
                    continue
                end

                cont=0.0
                if j == J
                    cont = ϕ(an, θ)
                else
                    cont = (1-πj)*V[j+1, k] + πj*ϕ(an, θ)
                end

                val = u_flow + β*cont
                if val > V_best
                    V_best = val
                    cbest = cj
                    anbest = an
                end
            end

            V[j, i] = V_best
            cpol[j, i] = cbest
            apol[j, i] = anbest
        end
    end

    return V, cpol, apol, agrid
end

function simulate_lifecycle(J::Int=60, a1::Float64=0.0, ȳ::Float64=1.0, θ::Float64=0.5)
    _, _, apol, agrid = solve_model(J, ȳ, θ)
    
    a_path = zeros(Float64, J)
    c_path = zeros(Float64, J)
    y_path = zeros(Float64, J)
    s_path = zeros(Float64, J)

    a_path[1] = a1
    for j in 1:J
        # Find closest asset grid index
        a_now = a_path[j]
        i = searchsortedlast(agrid, a_now)
        i = clamp(i, 1, length(agrid))

        anext = apol[j, i]
        cnow = c(a_now, anext, j, ȳ)

        c_path[j] = cnow
        y_path[j] = y(j, ȳ)
        s_path[j] = anext - a_now

        if j < J
            a_path[j+1] = anext
        end
    end

    return a_path, c_path, y_path, s_path
end

V, cpol, apol, agrid = solve_model(1.0, 0.5)


println("Is V defined for all j: ", all(isfinite, V[J, :]))
println("Are all consumption values non-negative: ", all(cpol .>= 0))
println("Are all asset policy values non-negative: ", all(apol .>= 0))
println("Are al asset policy values within bounds: ", all(apol[j, i] .<= (1+r)*agrid[i] + y(j) for j in 1:J, i in 1:500))

# ===== Life Cycle Sim Plot Helper =====
function plot_lifecycle(J::Int, 
                        a_path::Vector{Float64}, 
                        c_path::Vector{Float64}, 
                        y_path::Vector{Float64}, 
                        s_path::Vector{Float64}
                        ;
                        show::Bool=true,
                        saveloc::Union{Nothing, String}=nothing
                    )
    p = plot(
        layout = (2, 2),
        size=(800, 600),
    )
    plot!(p[1, 1], 1:J, c_path, 
        title = "Consumption over Lifecycle", 
        xlabel = L"Age ($j$)", 
        ylabel = L"Consumption $c_j$",
        legend = false,
    )

    plot!(p[1, 2], 1:J, y_path, 
        title = "Income over Lifecycle", 
        xlabel = L"Age ($j$)", 
        ylabel = L"Income $y_j$",
        legend = false,
    )
    plot!(p[2, 1], 1:J, a_path, 
        title = "Assets over Lifecycle", 
        xlabel = L"Age ($j$)", 
        ylabel = L"Assets $a_j$",
        legend = false,
    )
    plot!(p[2, 2], 1:J, s_path, 
        title = "Savings over Lifecycle", 
        xlabel = L"Age ($j$)", 
        ylabel = L"Savings $s_j$",
        legend = false,
    )
    if saveloc !== nothing
        savefig(p, saveloc)
    end

    if show
        display(p)
    end

end


# ===== Tasks =====

# Q3
ages = [20, 30, 40, 50, 60]

# 3.i Consumption
p1 = Plots.plot(
    title = L"Consumption policy $c_j(a)$",
    xlabel = L"$a$",
    ylabel = L"$c$",
    legend = :best
)
for j in ages
    plot!(p1, agrid, cpol[j, :], label = "j = $j")
end
# savefig(p1, "PS3/figure/p3_consumption_policy.png")
display(p1)

# 3.ii Savings
p2 = plot(
    title = L"Savings policy $a'_j(a)$",
    xlabel = L"$a$",
    ylabel = L"$a'$",
    legend = :best
)
for j in ages
    plot!(p2, agrid, apol[j, :], label = "j = $j")
end
# savefig(p2, "PS3/figure/p3_savings_policy.png")
display(p2)

# 3.iii Value Function
p3 = plot(
    title = L"Value function $V_j(a)$",
    xlabel = L"$a$",
    ylabel = L"$V$",
    legend = :best
)

for j in ages
    plot!(p3, agrid, V[j, :], label = "j = $j")
end
# savefig(p3, "PS3/figure/p3_value_function.png")
display(p3)

# Q4
a_path, c_path, y_path, s_path = simulate_lifecycle(J, 0.0, 1.0, 0.5)

# subplots for lifecycle simulation
plot_lifecycle(J, a_path, c_path, y_path, s_path;
                # saveloc="PS3/figure/p4_lifecycle_simulation.png",
                show=true
            )

# Q5
inc_levels = [0.5, 1.0, 2.0]
res_t05 = [
    simulate_lifecycle(J, 0.0, ȳ, 0.5) for ȳ in inc_levels
]

for (i, ȳ) in enumerate(inc_levels)
    a_path, c_path, y_path, s_path = res_t05[i]
    plot_lifecycle(J, a_path, c_path, y_path, s_path;
                    # saveloc="PS3/figure/p5_lifecycle_simulation_ybar_$(ȳ).png",
                    show=true
                )
end

# Q6
res_t0 = [
    simulate_lifecycle(J, 0.0, ȳ, 0.0) for ȳ in inc_levels
]

# Asset comp θ=0.5 and θ=0.0
p = plot(
    layout = (1, 2),
    size=(900, 400),
    plot_title = "Asset Paths for Different Income Levels",
)

for (i, ȳ) in enumerate(inc_levels)
    a_path_t05, _, _, _ = res_t05[i]
    a_path_t0, _, _, _ = res_t0[i]

    println("Peak Wealth for ȳ=$ȳ at θ=0.5: ", round(maximum(a_path_t05), digits=3))
    println("Peak Wealth for ȳ=$ȳ at θ=0.0: ", round(maximum(a_path_t0), digits=3))
    println()

    plot!(p[1], 1:J, a_path_t05, label = "ȳ = $ȳ", title = L"\theta = 0.5", xlabel = L"Age ($j$)", ylabel = L"Assets $a_j$")
    plot!(p[2], 1:J, a_path_t0, label = "ȳ = $ȳ", title = L"\theta = 0.0", xlabel = L"Age ($j$)", ylabel = L"Assets $a_j$")
end
savefig(p, "PS3/figure/p6_asset_path_comparison.png")
display(p)

inequality_nbequest = maximum(res_t0[3][1]) - maximum(res_t0[1][1])
inequality_bequest = maximum(res_t05[3][1]) - maximum(res_t05[1][1])

println("Inequality in Peak Wealth (θ=0.0): ", round(inequality_nbequest, digits=3))
println("Inequality in Peak Wealth (θ=0.5): ", round(inequality_bequest, digits=3))

# 6.d Effect of Bequest Motive
#
# A bequest motive creates an extra incentive to save by amplifying the utility of passing down assets.
# This motive takes a secondary role compared to consumption. Therefore, poorer agents who have to 
# consume most of their income are affected to a lesser degree by this motive.
#
# By extension, richer agents have more leeway to save and preserve for bequests.
# Thus, the presence of bequest motives increase the income inequality by affecting the wealthier agents.
# Both the peaks in asset paths, as well as the inequality figures confirm this.
#
# If we observe the plots, the difference in asset paths become much more pronounced for higher income Levels
# while the lowest earners show negligible difference.
#
# This phenomenon directly shows how bequests are "luxury goods".
# The easier one can cover consumption needs, the more one can allocate resources to bequests.