###############################################################################
################################ NUMERICS.JL ##################################

########### This script defines useful numerical functions to solve ###########
###################### the benchmark ProgTax(2025) model ######################

###############################################################################

using LinearAlgebra
using Roots


# Solving non-linear equation for consumption given by Feldstein tax specification
# Equation type: 2c - λ_c*c^(1 - τ_c) = k
function find_c_feldstein(k, lambda_c, tau_c)
    # Define function
    f = c -> 2c - lambda_c * c ^ (1 - tau_c) - k

    # Find solution
    # c_star = find_zero(f, (0,1), Bisection()) #Benchmark
    c_star = find_zero(f, 0.5)

    return c_star
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
    new_matrix = matrix .* ones(Float64, Tuple(levels))

    return new_matrix
end

# Starting from a vector, create an Array of N dimension
# with the nth dimension collecting the vector values
# Used in matrix operations in combination with ExpandMatrix
function Vector2NDimMatrix(x::AbstractVector, Ndims)
    return reshape(x, Tuple(push!(ones(Int64, Ndims), length(x))))
end
