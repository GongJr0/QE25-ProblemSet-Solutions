using QuadGK,Roots,Plots

#1
function  foc_integral(w,W,Rf,γ,μ,σ)
    
    fr(r)=1/(r*σ*sqrt(2*pi))*exp(-((log(r)-μ)^2)/(2*σ^2))
    
    focintegral(r,w,W,Rf,γ)=(r-Rf)*(W*(w*r+(1-w)*Rf))^(-γ)*fr(r)
    
    numer_evaluated,_=quadgk(r -> focintegral(r,w,W,Rf,γ), 0, Inf)
    
    return numer_evaluated
end

#2
function test_2(;
    γ=0,            #this was the risk aversion
    w=0.5,          #how much of wealth is invested in the risky one
    W=4,            #starting wealth
    Rf=0.1,         #return on the risk free asset
    μ=0,
    σ=1)
    
    num_eva= foc_integral(w,W,Rf,γ,μ,σ)
    
    analytic_value=exp(μ+(σ^2)/2)-Rf
    
    if isapprox(num_eva, analytic_value, atol=1e-8)
        println("Its a match!")
        println("Our result: ",num_eva)
        println("Exp result: ", analytic_value)
    else
        println("Ouugh 🦭")
    end
end

test_2()

#3
function optimal_portfolio(W,Rf,γ,μ,σ)
    F(w)=foc_integral(w,W,Rf,γ,μ,σ)

    a,b=-1.0,1.0
    for _ in 1:20
        if sign(F(a)) != sign(F(b))
            return find_zero(F, (a,b), Bisection())     #kinda long so prolly a better way to do it
        end
        a=a*2
        b=b*2
    end
end

#4
W=1
Rf=1.02
γ=3
μ=0.05
σ=0.10

w_4=optimal_portfolio(W,Rf,γ,μ,σ)
println("The optimal share of w_o is: ",w_4)

#5
graph_γ=range(0.1,10)

function grapher(W,Rf,graph_γ,μ,σ)
    w_optimal=[]

    for γ in graph_γ
        w_o=optimal_portfolio(W,Rf,γ,μ,σ)
        push!(w_optimal,w_o)
    end
    plot(graph_γ, w_optimal,linewidth=2, xlabel="relative risk aversion",ylabel="Optimal share of risky assets",Title="Impact of risk aversion on the composition of a portfolio")                     #need to add the labels later on
end
grapher(1,1.02,graph_γ,0.05,0.10)