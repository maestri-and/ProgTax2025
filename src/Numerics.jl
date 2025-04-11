###############################################################################
################################ NUMERICS.JL ##################################

########### This script defines useful numerical functions to solve ###########
###################### the benchmark ProgTax(2025) model ######################

###############################################################################

using LinearAlgebra
using Roots


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


# Function to extract stable distribution from transition matrix 

function find_stable_dist(transition_matrix; max_iter = 5000)
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
            @error("No solution found after $iter iterations")
        end
        stable_dist .= temp
    end
end

