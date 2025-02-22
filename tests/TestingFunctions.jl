###############################################################################
############################ TESTINGFUNCTIONS.JL ##############################

######### This script defines testing functions that are used in the  #########
###### unit root tests and in-line for the benchmark ProgTax(2025) model ######

###############################################################################

using Test
using DataFrames
include("../src/model_parameters.jl")
include("../src/households.jl")
include("../src/SolvingFunctions.jl")

###############################################################################
###############################  IN-CODE TESTS ################################
###############################################################################


function test_budget_constraint()
    # This function tests that the budget constraint holds after the creation
    # of the consumption matrix
    println("Test #1: Testing that budget constraint holds...")

    # Pick two random initial states 
    rand_a_i = rand(1:N_a)
    rand_rho_i = rand(1:N_rho)
    rand_l_i = rand(1:N_l)
    rand_a_prime_i = rand(1:N_a)

    rand_a = a_grid[rand_a_i]
    rand_rho = rho_grid[rand_rho_i]
    rand_a_prime = a_grid[rand_a_prime_i]
    rand_l = l_grid[rand_l_i]

    # println("Testing for the following random values:")
    # println("l = $rand_l [$rand_l_i]")
    # println("rho = $rand_rho [$rand_rho_i]")
    # println("a = $rand_a [$rand_a_i]")
    # println("a' = $rand_a_prime [$rand_a_prime_i]")

    # Compute net income
    rand_y = rand_l * rand_rho * w
    rand_labor_tax = hh_labor_taxes[rand_l_i, rand_rho_i]
    rand_return = (1 + r) * rand_a

    # Compute consumption and consumption tax
    rand_c = hh_consumption[rand_l_i, rand_rho_i, rand_a_i, rand_a_prime_i]
    rand_cons_tax = hh_consumption_tax[rand_l_i, rand_rho_i, rand_a_i, rand_a_prime_i]

    # Check that budget constraint holds
    rand_income = rand_y + rand_return 
    rand_taxes = rand_labor_tax + rand_cons_tax
    rand_expenditure = rand_c + rand_a_prime

    # Print output of check
    if (abs(rand_income - rand_taxes - rand_expenditure) < 0.001 || rand_expenditure == -Inf)
        # Round values for printing 
        # rand_c_ = round(rand_c; digits = 3)
        # rand_cons_tax_ = round(rand_cons_tax; digits = 3)
        # rand_labor_tax_ = round(rand_labor_tax; digits = 3)
        # rand_y_ = round(rand_y; digits = 3)
        # rand_return_ = round(rand_return; digits = 3)
        # rand_a_prime_ = round(rand_a_prime; digits = 3)

        # Print
        println("Test passed!")
        # println("c + T(c) = y - T(y) + (1 + r)a - a'")
        # println("$rand_c_ + $rand_cons_tax_ = $rand_y_ - $rand_labor_tax_ + $rand_return_ - $rand_a_prime_")
    else 
        error("Test failed! Budget constraint does not hold.\n" *
              "Income: $rand_income\n" *
              "Taxes: $rand_taxes\n" *
              "Expenditure: $rand_expenditure")
    end

end


function test_optimal_budget_constraint()
    # This function tests that the budget constraint holds after the 
    # Value Function Iteration 
    println("Test #1: Testing that budget constraint holds...")

    # Pick two random initial states 
    rand_a_i = rand(1:N_a)
    rand_rho_i = rand(1:N_rho)

    rand_a = a_grid[rand_a_i]
    rand_rho = rho_grid[rand_rho_i]

    println("Testing for the following random state:")
    println("rho = $rand_rho [$rand_rho_i]")
    println("a = $rand_a [$rand_a_i]")

    # Extract optimal choices
    rand_opt_a_prime_i = policy_a_index[rand_rho_i, rand_a_i]
    rand_opt_a_prime = a_grid[rand_opt_a_prime_i]

    rand_opt_l_i = policy_l_index[rand_rho_i, rand_a_i]
    rand_opt_l = l_grid[rand_opt_l_i]

    println("Optimal choices found: ")
    println("l = $rand_opt_l [$rand_opt_l_i]")
    println("a' = $rand_opt_a_prime [$rand_opt_a_prime_i]")

    # Compute net income
    rand_y = rand_opt_l * rand_rho * w
    println("Labor income: $rand_y")
    rand_labor_tax = hh_labor_taxes[rand_opt_l_i, rand_rho_i]
    rand_return = (1 + r) * rand_a
    println("Asset income: $rand_return")

    # Compute consumption and consumption tax
    rand_c = hh_consumption[rand_opt_l_i, rand_rho_i, rand_a_i, rand_opt_a_prime_i]
    rand_cons_tax = hh_consumption_tax[rand_opt_l_i, rand_rho_i, rand_a_i, rand_opt_a_prime_i]

    println("Consumption level: $rand_c")
    println("Consumption taxes: $rand_cons_tax")

    # Check that budget constraint holds
    rand_income = rand_y + rand_return 
    rand_taxes = rand_labor_tax + rand_cons_tax
    rand_expenditure = rand_c + rand_opt_a_prime

    rand_income - rand_taxes == rand_expenditure
    

    # Print output of check
    if (abs(rand_income - rand_taxes - rand_expenditure) < 0.001)
        println("Test passed!")
    else 
        error("Test failed! Budget constraint does not hold.\n" *
                "Income: $rand_income\n" *
                "Taxes: $rand_taxes\n" *
                "Expenditure: $rand_expenditure")
    end
end


###############################################################################
################################# OTHER TESTS #################################
###############################################################################

# Plotting dummy 3D utility function 


# Function to plot the utility function and return the utility matrix and maximum point
function plot_utility_function(rra, phi, frisch; normalise = false, c_range = (0.1, 5.0), l_range = (0.1, 5.0), num_points = 100)
    # Generate a range of consumption and labor values
    c_values = range(c_range..., length = num_points)
    l_values = range(l_range..., length = num_points)

    # Create a grid of consumption and labor values
    utility_matrix = [get_utility_hh(c, l, rra, phi, frisch, normalise = normalise) for c in c_values, l in l_values]

    # Transpose the utility matrix for correct plotting
    utility_matrix = utility_matrix'

    # Plot the utility function
    p = plot(c_values, l_values, utility_matrix, st = :surface, xlabel = "Consumption (c)", ylabel = "Labor (l)", zlabel = "Utility", title = "Utility Function")

    # Find the maximum utility value and its coordinates
    max_utility = maximum(utility_matrix)
    max_index = argmax(utility_matrix)
    max_c = c_values[max_index[1]]
    max_l = l_values[max_index[2]]

    println("Maximum utility: $max_utility at (c = $max_c, l = $max_l)")

    # Return utility matrix and plot
    return utility_matrix, p, (max_utility, max_c, max_l)
end

# # Example usage
# ut_matrix, utility_plot, max_point = plot_utility_function(2.0, 1.0, 0.5; normalise = false)

# utility_plot
# max_point

# Function to plot the utility function and return the utility matrix and maximum point
# Imposing budget constraint to hold for given a, rho, a'

function plot_utility_with_bc(rra, phi, frisch; a_i = 10, a_prime_i = 10, rho_i = 3, normalise = false, l_grid = l_grid)
    # Choose labor grid
    l_values = l_grid

    # Compute household taxes, consumption, and utility
    @elapsed _, hh_consumption, _, hh_utility = compute_hh_taxes_consumption_utility_(a_grid,
                                                                    N_a, rho_grid, l_values, w, r, Tau_y, Tau_c, taxes, hh_parameters)

    # Fix one level of a and a'
    c_values = hh_consumption[:, rho_i, a_i, a_prime_i]
    utility_values = hh_utility[:, rho_i, a_i, a_prime_i]

    # Plot consumption and labor
    p1 = plot(l_values, c_values, xlabel = "Labor (l)", ylabel = "Consumption (c)", title = "Consumption and Utility vs. Labor - Fixed ρ, a and a'", label = "Consumption")

    # Plot utility and labor
    p2 = plot(l_values, utility_values, xlabel = "Labor (l)", ylabel = "Utility", label = "Utility", linecolor = :red)

    # Combine plots
    p = plot(p1, p2, layout = (2, 1), size = (800, 600))

    # Find the maximum utility value and its coordinates
    max_utility = maximum(utility_values)
    max_index = argmax(utility_values)
    max_c = c_values[max_index]
    max_l = l_values[max_index]

    println("Maximum utility: $max_utility at (c = $max_c, l = $max_l)")

    # Return utility values and plot
    return utility_values, p, (max_utility, max_c, max_l)
end

# utility_values, p, (max_utility, max_c, max_l) = plot_utility_with_bc(2.0, 1.0, 0.5; a_i = 10, a_prime_i = 10, rho_i = 3, normalise = false, l_grid = l_grid)
# p #Display graphs
# savefig(p, "output/preliminary/utility_consumption_labor_budget_constraint.png")