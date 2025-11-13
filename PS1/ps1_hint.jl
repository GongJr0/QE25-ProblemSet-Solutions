using Distributions, Random


# define some Distributions
dist_exp = Exponential(0.3) # Exponential distribution with rate 0.3
dist_norm = Normal(2, 1)    # Normal distribution with mean 2 and standard deviation 1


# i will now draw 10 samples from each distribution

draw_exp = rand(dist_exp, 10)
draw_norm = rand(dist_norm, 10)


# observe that i get different numbers every time - just exectute these lines several times
# this is because the numbers are drawn randomly

# I can also set the seed of the random number generator to get the same numbers every time
draw_exp_fix = rand(MersenneTwister(222),dist_exp, 10)
draw_norm_fix = rand(MersenneTwister(222),dist_norm, 10)


# often it is useful to fix the seed at the beginning of your scropt to make sure that everyone gets the same results when executing your code
Random.seed!(1111)  # that's how you do it


