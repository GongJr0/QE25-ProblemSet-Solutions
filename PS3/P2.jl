using Random, Distributions, Statistics, Plots, LaTeXStrings


# ====== Problem Setup (Params, Structs, Funcs) ======
struct Orchid
    DEAD::Int8  # Dead
    N_BLM::Int8  # Not Blooming
    BLM::Int8  # Blooming
end
O = Orchid(-1, 0, 1)

# Struct val → idx mapper
o_idx(state::Int8) = state == O.DEAD ? 1 : state == O.N_BLM ? 2 : 3


struct Action
    W::Int8  # Wait
    A::Int8  # Apply
end
ACT = Action(0, 1)

# Parameters
T    = 20
Smax = 10
λ    = 0.05
δ    = 0.25
ρ    = 5.0
c0   = 0.5
c1   = 0.5
c2   = 0.1
α    = 5.0
ψ    = 10.0
κ    = 0.5
θ    = 2.0
ω    = 2.5

# r(O, t)
function r(O_STATE::Int8, t::Int)
    R_POLY = -(c0+ c1*t + c2*t^2)

    if O_STATE == O.DEAD
        return R_POLY - ψ

    elseif O_STATE == O.N_BLM
        return R_POLY

    elseif O_STATE == O.BLM
        return α
    end
end

# Φ(O, p)
function Φ(O_FINAL::Int8, p::Float64)
    if O_FINAL == O.DEAD
        return -ω*p

    elseif O_FINAL == O.N_BLM
        return -κ*p
    
    elseif O_FINAL == O.BLM
        return θ*p
    end
end

#Truncated Pois
function TPois(λ::Float64, K::Int)
    d = Poisson(λ)
    p = [pdf(d, k) for k in 0:K]
    tail = 1-sum(p) 
    p[end] += tail # Ensure total probability (∑P(K=k) = 1)
    return p
end


function solve_model(p; K=100)
    probs = TPois(ρ, K)
    V = zeros(T, 3, Smax+1) # V[t, o_idx, s]
    policy = fill(ACT.W, T, 3, Smax+1) # policy[t, o_idx, s]

    # V(Ot, St) [Terminal]
    for S in 0:Smax
        for O_STATE in (O.DEAD, O.N_BLM, O.BLM)
            V[T, o_idx(O_STATE), S+1] = r(O_STATE, T) + Φ(O_STATE, p)
        end
    end

    # V(Ot, St) [Backwards Recursion]
    for t in (T-1):-1:1
        for S in 0:Smax
            for O_STATE in (O.DEAD, O.BLM)
                # Don't change state
                V[t, o_idx(O_STATE), S+1] = r(O_STATE, t) + V[t+1, o_idx(O_STATE), S+1]
                policy[t, o_idx(O_STATE), S+1] = ACT.W
            end

            # O_STATE == O.N_BLM (Choose Action)
            W = λ * V[t+1, o_idx(O.BLM), S+1] + (1-λ) * V[t+1, o_idx(O.N_BLM), S+1]
            vW = r(O.N_BLM, t) + W

            A = 0.0
            for k in 0:K
                pk = probs[k+1]
                S_next = S + k
                if S_next > Smax
                    A += pk * V[t+1, o_idx(O.DEAD), Smax+1]
                else
                    A += pk * (
                        (λ+δ) * V[t+1, o_idx(O.BLM), S_next+1] 
                        +
                        (1 - λ - δ) * V[t+1, o_idx(O.N_BLM), S_next+1]
                    )
                end
            end
            vA = r(O.N_BLM, t) + A

            if vA > vW
                V[t, o_idx(O.N_BLM), S+1] = vA
                policy[t, o_idx(O.N_BLM), S+1] = ACT.A
            else
                V[t, o_idx(O.N_BLM), S+1] = vW
                policy[t, o_idx(O.N_BLM), S+1] = ACT.W
            end
        end
    end

    return V, policy
end

function one_ahead_sim(pol; K::Int=100, seed::Int=0)
    Random.seed!(seed)
    probs = TPois(ρ, K)
    d = Categorical(probs)  # For sampling k
    O_STATE = O.N_BLM
    S = 0
    fert = 0

    is_bloomed = falses(T)
    is_bloomed[1] = (O_STATE == O.BLM)

    for t in 1:T-1
        if O_STATE == O.N_BLM
            a = pol[t, o_idx(O_STATE), S+1]
            if a == ACT.W
                if rand() < λ
                    O_STATE = O.BLM
                end
            elseif a == ACT.A
                fert += 1
                k = rand(d) - 1 # k = list_idx - 1 
                if S + k > Smax
                    O_STATE = O.DEAD
                else
                    S += k
                    if rand() < (λ + δ)
                        O_STATE = O.BLM
                    end
                end
            end
        end
        is_bloomed[t+1] = (O_STATE == O.BLM)
    end
    return O_STATE, fert, is_bloomed
end

# ====== Tasks ======
p = 100.0
V, pol = solve_model(p, K=100)

# (a) Optimal Action (t=1, O=N_BLM, S=0)
println("Opmtimal Action at (t=1, O=N_BLM, S=0): ", 
        pol[1, o_idx(O.N_BLM), 1] == ACT.A ? "Apply Fertilizer" : "Wait"
        )
println()

# (b) Policy plot as function of {S, t} at O=N_BLM
t_arr = [1, 5, 10, 15, 19]
S_arr = 0:Smax
p = plot(
    title=L"Opmtimal Policy at Stress Level $S$",
    xlabel=L"$S$",
    ylabel="Action (0=Wait, 1=Apply Fertilizer)",
)

for t in t_arr
    actions = [pol[t, o_idx(O.N_BLM), S+1] for S in S_arr]
    plot!(p, S_arr, actions, label="t=$t", marker=:circle)
end

# savefig(p, "PS3/figure/p2_policy.png")
display(p)

# (c) Expected total utility (t=1, O=N_BLM, S=0)
println("Expected Total Utility at (t=1, O=N_BLM, S=0): ",
        V[1, o_idx(O.N_BLM), 1]
        )
println()

# (d) One-ahead simulation
Nsim = 1000
seed = 0

finals = Vector{Int8}(undef, Nsim)
ferts = Vector{Int}(undef, Nsim)
bloom_p_t = zeros(T)
for i in 1:Nsim
    final, fert, is_bloomed = one_ahead_sim(pol, K=100, seed=seed+i)
    finals[i] = final
    ferts[i] = fert
    bloom_p_t .+= is_bloomed
end

bloom_p_t ./= Nsim

frac_dead = sum(finals .== O.DEAD) / Nsim
frac_nblm = sum(finals .== O.N_BLM) / Nsim
frac_blm = sum(finals .== O.BLM) / Nsim

println("Sanity Check (n_blm / N = P(O_T = BLM)): ", frac_blm == bloom_p_t[end])
println("n_dead / N = ", frac_dead)
println("n_nblm / N = ", frac_nblm)
println("n_blm / N = ", frac_blm)
println()

# (e) Average fertilizer applications
avg_fert = mean(ferts)
println("Average Fertilizer Applications: ", avg_fert)
println()

# (f) Plot P(Bloomed) over time
p2 = plot(
    1:T, bloom_p_t,

    title="Probability of Blooming Over Time",
    xlabel=L"Time ($t$)",
    ylabel=L"$P(O_t = \operatorname{BLM})$",
    legend=false,
)
# savefig(p2, "PS3/figure/p2_bloom_probt.png")
display(p2)

# (g) V1(0, 0) for p ∈ [50, 200]

p_arr = 50:200
Vs = zeros(length(p_arr))
for (i, p) in enumerate(p_arr)
    Vp, _ = solve_model(Float64(p), K=100)
    Vs[i] = Vp[1, o_idx(O.N_BLM), 1]
end

p3 = plot(
    p_arr, Vs,

    title=L"$V_1(0, \, 0)$ as Function of $p$",
    xlabel=L"$p$",
    ylabel=L"$V_1(0, \ 0)$",
    legend=false,
)
hline!(p3, [0.0], linestyle=:dash) # mark 0
# savefig(p3, "PS3/figure/p2_V1_vs_p.png")
display(p3)

idx_first_pos = findfirst(Vs .> 0.0)
println("First p where V1(0, 0) > 0: ", p_arr[idx_first_pos])