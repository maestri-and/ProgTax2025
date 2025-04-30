###############################################################################
################################ NUMERICS.JL ##################################

########### This script defines useful numerical functions to solve ###########
###################### the benchmark ProgTax(2025) model ######################

###############################################################################

using LinearAlgebra
using Roots
using Interpolations
using CairoMakie

###############################################################################
###############################################################################
############################### 1. ROOT FINDING ###############################
###############################################################################
###############################################################################


# Solving non-linear equation for consumption given by Feldstein tax specification
# Equation type: 2c - λ_c*c^(1 - τ_c) = k
# This function allows for negative values of k but positive values of consumption
# i.e., it allows for government redistribution through consumption tax
# if the No-tax area upper bound is not specified (notax_upper)
# If notax_upper is specified, the function will assume zero taxes up to that point

function find_c_feldstein(k, lambda_c, tau_c; notax_upper=nothing)
    # Define the function based on the notax_upper argument
    if notax_upper !== nothing && k <= notax_upper
        # Use the simplified function if notax_upper is provided
        # If there is a tax-exemption area, impose T(c) to be zero in that region
        f = c -> c - k
    else
        # Use the original function otherwise
        f = c -> 2c - lambda_c * c^(1 - tau_c) - k
    end

    # Pre-allocate c_star with a default value
    c_star = 0.0

    # Try to find the solution
    try
        # Find solution, if any
        c_star = find_zero(f, 0.5) #0.5 Initial guess, adjustable
    catch e
        if isa(e, DomainError) | isa(e, Roots.ConvergenceFailed)
            # Handle DomainError by returning -Inf
            return -Inf
        else
            # Rethrow other exceptions
            throw(e)
        end
    end

    return c_star
end

# TBM
# Can be more efficient than find_c_feldstein in case there are a lot of non-feasible choices
function imp_find_c_feldstein(k, lambda_c, tau_c; notax_upper=nothing)
    # Check the tax-exemption condition upfront to avoid unnecessary allocations
    if notax_upper !== nothing && k <= notax_upper
        return k  # Directly return k if in the tax-exemption zone
    end
    
    # Define the function outside the try block to reduce overhead
    f(c) = 2c - lambda_c * c^(1 - tau_c) - k

    # Use try-catch only for the root finding
    try
        # Find solution with an initial guess suitable for the expected range
        return find_zero(f, 0.5)
    catch e
        # Directly check for DomainError
        return isa(e, DomainError) ? -Inf : throw(e)
    end
end

# Expand matrix by one dimension
# This function expands a matrix by one dimension, 
# preparing it for further matrix operations 
function ExpandMatrix(matrix::AbstractArray, new_dim_length::Int64)
    # Read dimension of current matrix
    mat_size = collect(size(matrix))

    # Add one extra dimension to the matrix
    matrix = reshape(matrix, Tuple(push!(copy(mat_size), 1)))

    # Add as many levels as needed for the new dimension
    levels = push!(ones(Int64, length(mat_size)), new_dim_length)

    # Expand matrix
    matrix = matrix .* ones(Float64, Tuple(levels))

    return matrix
end

# Starting from a vector, create an Array of N dimension
# with the nth dimension collecting the vector values
# Used in matrix operations in combination with ExpandMatrix
function Vector2NDimMatrix(x::AbstractVector, Ndims)
    return reshape(x, Tuple(push!(ones(Int64, Ndims), length(x))))
end


# Hybrid root-finding function that switches across different algorithms to
# find the root for solving the budget constraint wrt c using l(c)
# implied by the Labor FOC
# Sharp discontinuity - f(c) tends to minus infinity for c going to zero
# from both left and right - makes this an ill-conditioned root-finding problem

# Strategy: use Secant method, if convergence fails or result found is negative, try bisection 
function robust_root_FOC(f, lower::Float64, upper::Float64; secant_guess=1.0)
    x_star = NaN
    try
        x_star = find_zero(f, secant_guess, Order1())
        if x_star < 0
            # If Secant succeeds but returns a negative root, try Bisection
            x_star = find_zero(f, (lower, upper), Bisection())
        end
    catch
        # If Secant fails, try Bisection
        x_star = find_zero(f, (lower, upper), Bisection())
    end
    return x_star
end


###############################################################################
###############################################################################
########################### 2. STATIONARY MATRICES ############################
###############################################################################
###############################################################################


# Function to extract stable distribution from transition matrix 

function find_stable_dist(transition_matrix; max_iter = 10000)
    # Extract dimensions of transition matrix to build new array
    stable_dist = ones(size(transition_matrix)[1])/size(transition_matrix)[1]
    temp = similar(stable_dist)
    p = transpose(transition_matrix)
    # Iterate until convergence
    for iter in 1:max_iter
        temp .= p * stable_dist
        if maximum(abs, temp .- stable_dist) < 1e-9
            # @info("Stable distribution: found solution after $iter iterations")
            return temp
        elseif iter == max_iter
            error("No solution found after $iter iterations")
            @error("No solution found after $iter iterations")
        end
        stable_dist .= temp
    end
end


###############################################################################
###############################################################################
############################# 3. EFFECTIVE TAXES ##############################
###############################################################################
###############################################################################


"""
    compute_effective_tax(brackets::Vector{Tuple{Tuple{Float64, Float64}, Float64}};
                          output::Symbol = :rate,
                          plot::Bool = false,
                          max_income::Float64 = 100_000,
                          step::Float64 = 100.0)

Compute the effective tax rate or total tax amount for a given progressive tax schedule.

# Arguments
- `brackets`: A vector of tuples, each containing:
    - A tuple `(lower_bound, upper_bound)` representing the income range (upper bound exclusive).
    - A `rate` representing the tax rate for that range.

# Keyword Arguments
- `output`: Symbol indicating the desired output:
    - `:rate` for effective tax rate.
    - `:taxes` for total tax amount.
- `plot`: Boolean indicating whether to display a plot.
- `max_income`: Maximum income level to consider (default: 100,000).
- `step`: Step size for income levels (default: 100.0).

# Returns
- An interpolation function mapping income to effective tax rate or total tax amount.
"""

function compute_effective_tax(brackets::Vector{Tuple{Tuple{Float64, Float64}, Float64}};
                               output::Symbol = :rate,
                               plot::Bool = false,
                               max_income::Float64 = 300000.00,
                               steps::Float64 = 1000.0,
                               graph_title = "Tax Schedule")   

    # Generate income levels
    incomes = 0.0:steps:max_income
    taxes = zeros(length(incomes))

    # Compute taxes for each income level
    for (i, income) in enumerate(incomes)
        tax = 0.0
        for ((lower, upper), rate) in brackets
            if income > lower
                taxable_income = min(income, upper) - lower
                tax += taxable_income * rate
            end
        end
        taxes[i] = tax
    end

    # Determine output
    if output == :rate
        effective_values = taxes ./ incomes
        effective_values[1] = 0.0  # Handle division by zero at income = 0
        ylabel = "Effective Tax Rate"
    elseif output == :taxes
        effective_values = taxes
        ylabel = "Total Tax Amount"
    else
        error("Invalid output type. Use :rate or :taxes.")
    end

    # Create interpolation function
    interp_func = LinearInterpolation(collect(incomes), effective_values, extrapolation_bc=Line())

    # Plot if requested
    if plot
        fig = CairoMakie.Figure(size = (800, 500))
        ax = CairoMakie.Axis(fig[1, 1], 
                             xlabel = "Gross Income", 
                             ylabel = ylabel, 
                             title = graph_title,
                             xtickformat = x -> string.(Int.(round.(x))),
                             ytickformat = y -> string.(round.(y; digits=2)))
        CairoMakie.lines!(ax, incomes, effective_values, color = :blue)
        display(fig)
    end

    return interp_func
end


###############################################################################
###############################################################################
############################### 4. CALIBRATION ################################
###############################################################################
###############################################################################

#-#-#-#-#-#-# Calibrating Feldstein function to target tax curve #-#-#-#-#-#-#

function minimise_tax_curve_distance(target_curve, functional_form;
                                     targeting_range)
    # Fit functional form to target curve by minimising sum of squared distances
end









