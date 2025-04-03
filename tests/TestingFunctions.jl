###############################################################################
############################ TESTINGFUNCTIONS.JL ##############################

######### This script defines testing functions that are used in the  #########
###### unit root tests and in-line for the benchmark ProgTax(2025) model ######

###############################################################################

using Test
using DataFrames
include("../src/Parameters.jl")
include("../src/Households.jl")
include("../src/SolvingFunctions.jl")

###############################################################################
###############################################################################
###############################  IN-CODE TESTS ################################
###############################################################################
###############################################################################

###############################################################################
#------------------------#  BUDGET CONSTRAINT TESTS #-------------------------#
###############################################################################

function test_budget_constraint()
    # This function tests that the budget constraint holds after the creation
    # of the consumption matrix
    @info("Test #1: Testing that budget constraint holds...")

    # Pick two random initial states 
    rand_a_i = rand(1:gpar.N_a)
    rand_rho_i = rand(1:gpar.N_rho)
    rand_l_i = rand(1:gpar.N_l)
    rand_a_prime_i = rand(1:gpar.N_a)

    rand_a = a_grid[rand_a_i]
    rand_rho = rho_grid[rand_rho_i]
    rand_a_prime = a_grid[rand_a_prime_i]
    rand_l = l_grid[rand_l_i]

    # @info("Testing for the following random values:")
    # @info("l = $rand_l [$rand_l_i]")
    # @info("rho = $rand_rho [$rand_rho_i]")
    # @info("a = $rand_a [$rand_a_i]")
    # @info("a' = $rand_a_prime [$rand_a_prime_i]")

    # Compute net income
    rand_y = rand_l * rand_rho * w
    rand_labor_tax = hh_labor_tax[rand_l_i, rand_rho_i]
    rand_return = (1 + (1 - taxes.tau_k)r) * rand_a

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
        @info("Test passed!")
        # @info("c + T(c) = y - T(y) + (1 + (1 - taxes.tau_k)r)a - a'")
        # @info("$rand_c_ + $rand_cons_tax_ = $rand_y_ - $rand_labor_tax_ + $rand_return_ - $rand_a_prime_")
    else 
        @error("Test failed! Budget constraint does not hold.\n" *
              "Income: $rand_income\n" *
              "Taxes: $rand_taxes\n" *
              "Expenditure: $rand_expenditure")
    end

end

# Deprecated
function test_optimal_budget_constraint()
    # This function tests that the budget constraint holds after the 
    # Value Function Iteration 
    @info("Test #1: Testing that budget constraint holds...")

    # Pick two random initial states 
    rand_a_i = rand(1:gpar.N_a)
    rand_rho_i = rand(1:gpar.N_rho)

    rand_a = a_grid[rand_a_i]
    rand_rho = rho_grid[rand_rho_i]

    @info("Testing for the following random state:")
    @info("rho = $rand_rho [$rand_rho_i]")
    @info("a = $rand_a [$rand_a_i]")

    # Extract optimal choices
    rand_opt_a_prime_i = policy_a_index[rand_rho_i, rand_a_i]
    rand_opt_a_prime = a_grid[rand_opt_a_prime_i]

    rand_opt_l_i = policy_l_index[rand_rho_i, rand_a_i]
    rand_opt_l = l_grid[rand_opt_l_i]

    @info("Optimal choices found: ")
    @info("l = $rand_opt_l [$rand_opt_l_i]")
    @info("a' = $rand_opt_a_prime [$rand_opt_a_prime_i]")

    # Compute net income
    rand_y = rand_opt_l * rand_rho * w
    @info("Labor income: $rand_y")
    rand_labor_tax = hh_labor_tax[rand_opt_l_i, rand_rho_i]
    rand_return = (1 + (1 - taxes.tau_k)r) * rand_a
    @info("Asset income: $rand_return")

    # Compute consumption and consumption tax
    rand_c = hh_consumption[rand_opt_l_i, rand_rho_i, rand_a_i, rand_opt_a_prime_i]
    rand_cons_tax = hh_consumption_tax[rand_opt_l_i, rand_rho_i, rand_a_i, rand_opt_a_prime_i]

    @info("Consumption level: $rand_c")
    @info("Consumption taxes: $rand_cons_tax")

    # Check that budget constraint holds
    rand_income = rand_y + rand_return 
    rand_taxes = rand_labor_tax + rand_cons_tax
    rand_expenditure = rand_c + rand_opt_a_prime

    rand_income - rand_taxes == rand_expenditure
    

    # Print output of check
    if (abs(rand_income - rand_taxes - rand_expenditure) < 0.001)
        @info("Test passed!")
    else 
        @error("Test failed! Budget constraint does not hold.\n" *
                "Income: $rand_income\n" *
                "Taxes: $rand_taxes\n" *
                "Expenditure: $rand_expenditure")
    end
end

function test_optimal_budget_constraint()
    # This function tests that the budget constraint holds after the 
    # Value Function Iteration 
    @info("Test #1: Testing that budget constraint holds...")

    # Pick two random initial states 
    rand_a_i = rand(1:gpar.N_a)
    rand_rho_i = rand(1:gpar.N_rho)

    rand_a = a_grid[rand_a_i]
    rand_rho = rho_grid[rand_rho_i]

    @info("Testing for the following random state:")
    @info("rho = $rand_rho [$rand_rho_i]")
    @info("a = $rand_a [$rand_a_i]")

    # Extract optimal choices
    rand_opt_a_prime = policy_a[rand_rho_i, rand_a_i]
    rand_opt_l = policy_l[rand_rho_i, rand_a_i]

    @info("Optimal choices found: ")
    @info("l = $rand_opt_l")
    @info("a' = $rand_opt_a_prime")

    # Compute net income
    rand_y = rand_opt_l * rand_rho * w
    @info("Labor income: $rand_y")
    rand_labor_tax = tax_hh(rand_y, taxes.lambda_y, taxes.tau_y)
    rand_return = (1 + (1 - taxes.tau_k)r) * rand_a
    @info("Asset income: $rand_return")

    # Compute consumption and consumption tax
    rand_cexp = get_Cexp(rand_rho, w, r, rand_opt_l, rand_a, rand_opt_a_prime, taxes)
    rand_c = cExp2cInt[rand_cexp]
    rand_cons_tax = rand_cexp - rand_c

    @info("Consumption level: $rand_c")
    @info("Consumption taxes: $rand_cons_tax")

    # Check that budget constraint holds
    rand_income = rand_y + rand_return 
    rand_taxes = rand_labor_tax + rand_cons_tax
    rand_expenditure = rand_c + rand_opt_a_prime

    rand_income - rand_taxes == rand_expenditure
    

    # Print output of check
    if (abs(rand_income - rand_taxes - rand_expenditure) < 0.001)
        @info("Test passed!")
    else 
        @error("Test failed! Budget constraint does not hold.\n" *
                "Income: $rand_income\n" *
                "Taxes: $rand_taxes\n" *
                "Expenditure: $rand_expenditure")
    end
end

###############################################################################
#----------------------------#  EFFICIENCY TESTS #----------------------------#
###############################################################################


# Function to perform benchmarking and save results
function benchmark_grid_size(filename::String, n::Int)
    open(filename, "w") do file
        for i in 1:n
            try
                # Benchmark the action
                result = @benchmark my_action($i)

                # Extract time and memory allocation
                time = minimum(result.times)
                allocations = result.memory

                # Write the results to the file
                write(file, "Iteration $i: Time = $time, Allocations = $allocations\n")
            catch e
                # If an error occurs, write the error message and stop
                write(file, "Error at iteration $i: $e\n")
                break
            end
        end
    end
end

# Run the benchmarking script
# benchmark_and_save("benchmark_output.txt", 1000)

###############################################################################
############################ INTERPOLATION TESTING ############################
###############################################################################


function evaluate_interpolation(data_points, interpolator, fixed_indices; figure=true)
    """
    Evaluates the interpolation accuracy for an arbitrary number of dimensions.

    Arguments:
    - data_points: A multi-dimensional array containing the true function values.
    - interpolator: A function that takes free variables and returns interpolated values.
    - fixed_indices: A tuple specifying which indices to fix in data_points (for slicing).
    - figure (optional, default=true): If true, plots a heatmap of absolute errors.

    Returns:
    - Prints max/mean absolute and relative errors.
    - Displays a heatmap if `figure=true` (only for 2D free variables).
    """
    
    # Convert fixed indices to integers (ensure valid indexing)
    fixed_indices = Tuple(round(Int, i) for i in fixed_indices)

    # Determine dimensions and slice fixed indices
    num_dims = ndims(data_points)
    free_dims = setdiff(1:num_dims, keys(fixed_indices))  # Free dimensions

    # Extract relevant grid sizes
    free_sizes = size(data_points)[free_dims]  # Sizes of free dimensions

    # Generate grid points (assuming a uniform grid from 1 to size in each free dim)
    free_grids = [collect(1:size) for size in free_sizes]

    # Create iterator over all free variable combinations
    grid_combinations = Iterators.product(free_grids...)

    # Compute interpolated values
    interpolated_values = [interpolator([fixed_indices..., free...]) for free in grid_combinations]

    # Reshape to match original grid structure
    interpolated_values = reshape(interpolated_values, free_sizes...)

    # Extract true values from data at fixed indices
    true_values = data_points[fixed_indices..., :, :]

    # Compute absolute and relative errors
    abs_errors = abs.(interpolated_values .- true_values)
    max_abs_error, mean_abs_error = maximum(abs_errors), mean(abs_errors)

    rel_errors = abs_errors ./ max.(abs.(true_values), 1e-10)  # Avoid division by zero
    max_rel_error, mean_rel_error = maximum(rel_errors), mean(rel_errors)

    # Print error metrics
    @info("Max Absolute Error: ", max_abs_error)
    @info("Mean Absolute Error: ", mean_abs_error)
    @info("Max Relative Error: ", max_rel_error)
    @info("Mean Relative Error: ", mean_rel_error)

    # Plot heatmap if 2D free variables (i.e., last two dimensions vary)
    if figure && length(free_dims) == 2
        heatmap(free_grids[1], free_grids[2], abs_errors, title="Interpolation Absolute Errors",
                xlabel="Dimension $(free_dims[1])", ylabel="Dimension $(free_dims[2])", colorbar_title="Error")
    end
end

###############################################################################
############################# VFI SOLUTION TESTS ##############################
###############################################################################

#------------------------# FEASIBILITY CONSTRAINTS #--------------------------#
# Given solution policy functions, ensure that consumption and labor are within
# feasibility boundaries - c >= 0, 0 <= l <= 1