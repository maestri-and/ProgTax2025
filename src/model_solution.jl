###############################################################################
############################## MODEL_SOLUTION.JL ##############################

############################# This script solves ##############################
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

using LinearAlgebra
using Distances
using Base.Threads
using DelimitedFiles
# using DataFrames
# using Plots

include("model_parameters.jl")
include("production_and_government.jl")
include("AuxiliaryFunctions.jl")
include("numerics.jl")
include("households.jl")
include("SolvingFunctions.jl")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Assets
a_grid = makeGrid(a_min, a_max, N_a)

# Labor
l_grid = makeGrid(l_min, l_max, N_l)

# Labor productivity - Defined in model_parameters.jl
# rho_grid = rho_grid

# Taxation parameters


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 2. VALUE FUNCTION ITERATION #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Temporary (TBM)
w = 1
r = 0.05
Tau_c = Tau_y = 0.136 # Temporary TBM

# @elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility(a_grid, 
#                                                                     N_a, rho_grid, l_grid, w, r, taxes, hh_parameters);

# Simplified version for one degree of progressivity of labor income and consumption taxes
@elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility_(a_grid, 
                                                                    N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters);
        

# Checks
hh_consumption[hh_consumption .< 0]
hh_utility[hh_utility .== -Inf]                                                                    
hh_consumption_tax[hh_consumption_tax .== 0]

# VFI
# Initialise value function arrays 
V_guess = (a_grid' .* rho_grid) / (maximum(a_grid) * maximum(rho_grid));
V_new = zeros(N_rho, N_a)

# Initialise policy function array
policy_a_index =zeros(Int64, N_l, N_rho, N_a)
tv = zeros(N_a) # Temporary vector to store intermediate values

@time for iter in 1:comp_params.vfi_max_iter
    for i_l in 1:N_l
        for i_rho in 1:N_rho
            for i_a in 1:N_a
                tv .= (hh_utility[i_l, i_rho, i_a, :]' + hh_parameters.beta * pi_rho[i_rho, :]' * V_guess[:, :])'
                (V_new[i_rho, i_a], policy_a_index[i_l, i_rho, i_a]) = findmax(tv[:]) # Find the maximum value and corresponding policy index
            end
        end
    end
    if maximum(abs, V_guess .- V_new) < comp_params.vfi_tol
        println("Found solution after $iter iterations")
        break
    elseif iter == comp_params.vfi_max_iter
        println("No solution found after $iter iterations")
    end
    V_guess .= copy(V_new)  # Update the guess with the new value function
    # err = maximum(abs, Vguess .- Vnew) # Calculate the error between the current and new value functions
    #println("#iter = $iter, εᵥ = $err") # Print the iteration number and error
end 

##### PARALLELISED ####

# Store the VFI guess 
SaveMatrix(V_guess, "output/preliminary/V_guess_matrix.txt")
V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")


