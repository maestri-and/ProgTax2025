###############################################################################
################################## TESTS.JL ###################################

############## This script runs tests for functions used in the ###############
###################### the benchmark ProgTax(2025) model ######################

###############################################################################

using Test

include("../src/production_and_government.jl")
include("../src/auxiliary_functions.jl")
include("../src/numerics.jl")



## Tests on production function computation
@testset "Production Function" begin
    labor_vector = [0.5, 0.4, 1]
    capital_vector = [0.6, 1, 1]

    # Note: with this specification,  labor in column, capital in row
    prod_matrix = cd_production.(1, 0.33, capital_vector, labor_vector')
    @test typeof(prod_matrix) == Matrix{Float64}
    @test prod_matrix[2, 3] ==  1
end

## Tests on auxiliary functions
@testset "Auxiliary functions" begin
    # Grid generation
    a_grid = makeGrid(1, 5, 1000)
    @test length(a_grid) == 1000
end

## Tests on numerical functions
@testset "Feldstein Root Finding" begin
    # Feldstein specification solution
    # Flat rate
    rate = 0.2
    lambda = 1 - rate
    k = 1
    solution = find_c_feldstein(k, lambda, 0)
    @test (1 + rate)solution == 1 

    # Progressive rate
    prog_rate = 0.3
    solution = find_c_feldstein(k, lambda, prog_rate)

    @test k == solution + solution - lambda * solution^(1 - prog_rate)

end

@testset "ExpandMatrix function" begin
    # Generate parameters to test
    test_mat = [1 1 1; 0 0 0]
    new_dim_len = 4 

    # Manually expand matrix
    me_test_mat = reshape(test_mat, 2, 3, 1)
    me_test_mat = me_test_mat .* ones(1, 1, new_dim_len)

    # Expand using function
    fun_test_mat = ExpandMatrix(test_mat, new_dim_len)
    @test me_test_mat == fun_test_mat 
end


using BenchmarkTools

A = rand(4, 5, 6)  # Example matrix
s = size(A)
tvec = [1, 2, 3]

@btime push!(copy(tvec), 1)
@btime vcat(tvec, 1)



