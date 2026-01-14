using Parameters, Plots, LinearAlgebra, QuantEcon, Statistics, StatsBase

#Task 1

#First im gonna do the setup like in vifi_practical so the tasks 1-3 might be a bit of a mess
Setup = @with_kw (
    h_grid_min=0.0,
    h_grid_max=2.0,
    h_grid_size=100,
    h_grid=collect(range(h_grid_min, h_grid_max, length=h_grid_size)),
    
    edu_grid=collect(0.0:0.01:1.0),
    
    ρ=0.98,                                                                 #persistance of the AR(1) process
    σ=0.15,                                                                 #this is std for innovation so ig its kinda like stdev of the AR(1) process for productivity
    n_w=7                                                                   #grid size to discretize the AR(1) process
)


Agent = @with_kw (
    β=0.95,
    γ=1.5,
    ψ=0.5,
    δ=0.1,
    α=0.1,
    fconstant=1.2,

    u = γ == 1 ? log : c->(c^(1-γ)-1)/(1-γ),
    u′ = c-> c.^(-γ), 
    u′_inv = c-> c.^(-1 / γ),
    f=h->min(h^α+0.1,fconstant)
)


setup = Setup()

ag=Agent()

#discretize AR(1)
discretized_y = rouwenhorst(setup.n_w,setup.ρ, setup.σ,0.0)
sum(discretized_y.p, dims=2)    #sums to 1 so i hope its allright

discretized_y.p                    #so this would be my transition 💊 matrix

#the merger to end all mergers
ag=merge(ag,(prob_trans = discretized_y.p,prod_grid = exp.(discretized_y.state_values),init_dist = [0.5,0.5,0,0,0]))

function next_h(h,e,δ)
    return h+e-δ
end

#ig i got to round the h
function round_h(h_grid,h_value)
    return argmin(abs.(h_grid.-h_value))
end

#the other stuff wont work cause infinity

function bellman_operator!(V,V_new,policy_edu,setup,ag)

    for ih in eachindex(setup.h_grid)
        for iw in eachindex(ag.prod_grid)
            h=setup.h_grid[ih]
            w=ag.prod_grid[iw]

            best_e=0.0
            max_poli=-Inf
            for e in setup.edu_grid
                c=w*ag.f(h)*(1-e)
                if c<=0
                    continue
                end
            
            u_current=ag.u(c)-ag.ψ*e
            h′=next_h(h,e,ag.δ)
            if h′<setup.h_grid_min||h′>setup.h_grid_max
                total=u_current-10^10
            else 
                ih′=round_h(setup.h_grid, h′)
                expected_value=0.0
                 for iw_next in eachindex(ag.prod_grid)
                    expected_value += ag.prob_trans[iw, iw_next] * V[ih′, iw_next]
                 end
                 total=u_current+ag.β*expected_value
                end
                if total>max_poli
                    max_poli=total
                    best_e=e
                end
            end
            V_new[ih, iw] = max_poli
            policy_edu[ih, iw] = best_e
        end
    end
end



#Task 3
function solver_vfi(setup, ag; max_iter=1000,tolerance=1e-6,verbose=true)
   
    n_h = length(setup.h_grid)
    n_w = length(ag.prod_grid)
    V=zeros(n_h,n_w)
    V_new=zeros(n_h,n_w)
    policy_edu=zeros(n_h, n_w)

    for iter in 1:max_iter
        bellman_operator!(V,V_new,policy_edu,setup,ag)
        diff=maximum(abs.(V_new - V))
        
        if diff<tolerance
        V.=V_new
        break
        end
        V.=V_new
        if iter == max_iter
        println("🦭  aaAAh blerggh AAAHHH bleehh eeehh ")
        end
    end
    return V, policy_edu
end

V, policy_edu = solver_vfi(setup, ag, verbose=true)
#Task 4

#a)
w_indices = [1, 4, 7]
pa = plot( xlabel="Human Capital (h)", ylabel="Education Choice (e)",legend=:topright)
for iw in w_indices
    w = ag.prod_grid[iw]
    plot!(pa, setup.h_grid, policy_edu[:, iw], linewidth=2)
end
pa

#b)
pb = plot(xlabel="Human Capital (h)",ylabel="Value",legend=:bottomright)
for iw in w_indices
    w = ag.prod_grid[iw]
    plot!(pb, setup.h_grid, V[:, iw], linewidth=2)
end
pb

#c)
consumption = zeros(size(policy_edu))
for ih in eachindex(setup.h_grid)
    for iw in eachindex(ag.prod_grid)
        h = setup.h_grid[ih]
        w = ag.prod_grid[iw]
        e = policy_edu[ih, iw]
        consumption[ih, iw] = w * ag.f(h) * (1 - e)
    end
end

pc = plot(xlabel="Human Capital (h)", ylabel="Consumption",legend=:bottomright)
for iw in w_indices
    w = ag.prod_grid[iw]
    plot!(pc, setup.h_grid, consumption[:, iw], linewidth=2)
end
pc


#Task 5
function policy(h, w, setup, ag, policy_edu)
    ih = round_h(setup.h_grid, h)
    iw = round_h(ag.prod_grid, w)
    return policy_edu[ih, iw]
end


function simulator(h_0,w_0,setup,ag,policy_edu,t=1000)
    T=t+100
    h_t=zeros(T)
    w_t=zeros(T)
    e_t=zeros(T)
    c_t=zeros(T)
    w_trans=zeros(Int,T)

    h_t[1]=h_0
    w_trans[1]=w_0
    w_t[1]=ag.prod_grid[w_0]

    for i in 1:(T-1)
        h=h_t[i]
        iw=w_trans[i]
        w=ag.prod_grid[iw]

        e=policy(h,w,setup,ag,policy_edu)
        c=w*ag.f(h)*(1-e)
    

        e_t[i]=e
        c_t[i]=c

        h′=next_h(h,e,ag.δ)
        if h′<setup.h_grid_min
            h_t[i+1]=setup.h_grid_min
            println("too small")
        elseif h′>setup.h_grid_max
            h_t[i+1]=setup.h_grid_max
            println("too big")
        else
            h_t[i+1]=h′
        end

        w_trans[i+1]=sample(1:setup.n_w, Weights(ag.prob_trans[iw,:]))
        w_t[i+1]=ag.prod_grid[w_trans[i+1]]
    end
    e_t[T]=policy(h_t[T],w_t[T],setup,ag,policy_edu)
    c_t[T]=w_t[T]*ag.f(h_t[T])*(1-e_t[T])
    return h_t, w_t, e_t, c_t
end

h_0=1.0
w_0=4

h_total=zeros(1000,5)
w_total=zeros(1000,5)
e_total=zeros(1000,5)
c_total=zeros(1000,5)

for i in 1:5
    h_sim,w_sim,e_sim,c_sim=simulator(h_0,w_0,setup,ag,policy_edu,1000)

    h_total[:,i]=h_sim[101:1100]
    w_total[:,i]=w_sim[101:1100]
    e_total[:,i]=e_sim[101:1100]
    c_total[:,i]=c_sim[101:1100]
end


post_burn=1:100
p5h = plot(xlabel="Period", ylabel="h", legend=:best)
for i in 1:5
    plot!(p5h, post_burn, h_total[post_burn, i], alpha=0.5, linewidth=2)
end
p5h

p5w = plot(xlabel="Period", ylabel="w", legend=:best)
for i in 1:5
    plot!(p5w, post_burn, w_total[post_burn, i], alpha=0.5, linewidth=2)
end
p5w

p5e = plot(xlabel="Period", ylabel="e", legend=:best)
for i in 1:5
    plot!(p5e, post_burn, e_total[post_burn, i], alpha=0.5, linewidth=2)
end
p5e

p5c = plot(xlabel="Period", ylabel="c", legend=:best)
for i in 1:5
    plot!(p5c,post_burn, c_total[post_burn, i], alpha=0.5, linewidth=2)
end
p5c

plot(p5h, p5w, p5e, p5c, layout=(2,2), size=(1200, 800))

#b) 
h_flat = vec(h_total)
w_flat = vec(w_total)
e_flat = vec(e_total)
c_flat = vec(c_total)

# Calculate labor earnings y_t = w_t * f(h_t) * (1 - e_t)
y_flat = zeros(length(h_flat))
for i in eachindex(h_flat)
    y_flat[i] = w_flat[i] * ag.f(h_flat[i]) * (1 - e_flat[i])
end
#1) correlation between labor earnings and education effort
corr_y_e = cor(y_flat, e_flat)
#The correlation is negative, which is somewhat understandable given that the effort needed to learn is increasing
#-persuing education has higher oportunity cost leading to decrease in time spent on education and increase in work time

#2) The correlation between the wage shock wt and education
corr_w_e = cor(w_flat, e_flat)
#once again we get negative correlation which, as wages increase the opportunity time of getting education increases
#(why go to school if you can earn more working at zabka)
