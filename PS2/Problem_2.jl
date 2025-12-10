using Distributions, Random, Statistics, Optim, Plots

#Problem 2
Random.seed!(2024)

#1
function datasymulator(T,ρ,p,σ_l,σ_h)
    log_y0=0.0
    t=T+100
    log_y=zeros(t)
    log_y[1]=log_y0

    for n in 1:(t-1)
        draw_bern=rand(Bernoulli(p))
        
        if 1 == draw_bern
            Ldist=Normal(0,σ_l)
            Ldraw=rand(Ldist)
            ϵ=Ldraw
        elseif 0 == draw_bern 
            Hdist=Normal(0,σ_h)
            Hdraw=rand(Hdist)
            ϵ=Hdraw
        else 
            println("ouugh my 🦭 son cant swim")
            ϵ=1
        end
        log_y[n+1]= ρ*log_y[n]+ϵ
    end
    return log_y[101:(T+100)]

end

sim_data=datasymulator(500,0.90,0.80,0.10,0.30)

#2
m_1=std(sim_data)
m_2=cor(sim_data[1:499],sim_data[2:500])
m_3=kurtosis(diff(sim_data))

#3
function simulate_model(θ,T,σ_l,σ_h)
    ρ,p=θ
    log_y0=0.0
    t=T+100
    log_y=zeros(t)
    log_y[1]=log_y0
    
    for n in 1:(t-1)
        draw_bern=rand(Bernoulli(p))
        
        if 1 == draw_bern
            Ldist=Normal(0,σ_l)
            Ldraw=rand(Ldist)
            ϵ=Ldraw
        else
            Hdist=Normal(0,σ_h)
            Hdraw=rand(Hdist)
            ϵ=Hdraw
        end
        log_y[n+1]= ρ*log_y[n]+ϵ
    end
    return log_y[101:end]               #ig thats whats required idk
end

#4
function ssm_objective(θ,observed_data,σ_l,σ_h,S=100)
    ρ,p=θ
    T=length(observed_data)
    
    ob_m_1=std(observed_data)
    ob_m_2=cor(observed_data[1:499],observed_data[2:500])
    ob_m_3=kurtosis(diff(observed_data))

    Tsim_m_1=0
    Tsim_m_2=0
    Tsim_m_3=0

    for n in 1:S
        sim=simulate_model(θ,T,σ_l,σ_h)
        
        sim_m_1=std(sim)
        sim_m_2=cor(sim[1:499],sim[2:500])
        sim_m_3=kurtosis(diff(sim))

        Tsim_m_1=Tsim_m_1+sim_m_1
        Tsim_m_2=Tsim_m_2+sim_m_2
        Tsim_m_3=Tsim_m_3+sim_m_3
    end

    Atsim_m_1=Tsim_m_1/S
    Atsim_m_2=Tsim_m_2/S
    Atsim_m_3=Tsim_m_3/S

    Q=(ob_m_1-Atsim_m_1)^2+(ob_m_2-Atsim_m_2)^2+(ob_m_3-Atsim_m_3)^2
    return Q
end

#5
S=100
θ_0=[0.85,0.70]
low_bound=[0.5,0.5]
high_bound=[0.99,0.95]
σ_l=0.10
σ_h=0.30

objective(θ)=ssm_objective(θ,sim_data,σ_l,σ_h,S)

optimized_objective=optimize(objective, low_bound, high_bound, θ_0,  ParticleSwarm(lower=low_bound, upper=high_bound, n_particles=50))
#it says failure but I pinky promise its somewhat good
θ_estim=Optim.minimizer(optimized_objective)
ρ_estim,p_estim=θ_estim

println("The estimated ρ is: ",ρ_estim)
println("The estimated p is: ",p_estim)

function error_mesurment(ρ_estim,p_estim)
    error_ρ=ρ_estim-0.90
    error_p=p_estim-0.80
    println("The ρ error is equal to ",error_ρ)
    println("The p error is equal to ",error_p)
end
error_mesurment(ρ_estim,p_estim)

#6
mod_sim=simulate_model(θ_estim,500,σ_l,σ_h)

#log income lines
plot_1=plot(1:200,sim_data[1:200],color=:blue,linewidth=2,xlabel="Time",ylabel="log income for the first 200 peroids",legend=:false)
plot!(plot_1, 1:200, mod_sim[1:200],color=:orange,linewidth=2)        

#histogram of diff
dlog_obs=diff(sim_data)
dlog_sim=diff(mod_sim)

plot_2=histogram(dlog_obs,bins=50,alpha=0.5,label="observed data",color=:blue,normalize=:pdf)
histogram!(plot_2,dlog_sim,bins=50,alpha=0.5,label="simulated data",color=:orange,normalize=:pdf)

#I'd say they do given how histograms shows somewhat similar distribution among beans