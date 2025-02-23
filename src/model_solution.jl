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
using Interpolations
using DelimitedFiles
# using DataFrames
using Plots
using BenchmarkTools


include("model_parameters.jl")
include("production_and_government.jl")
include("AuxiliaryFunctions.jl")
include("numerics.jl")
include("households.jl")
include("SolvingFunctions.jl")
include("../tests/TestingFunctions.jl")


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


#### COMPUTING TAXES, CONSUMPTION AND UTILITY FOR EACH STATE-CHOICE POINT #####

# Original version - including also tax progressivity rates
# @elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility(a_grid, 
#                                                                     N_a, rho_grid, l_grid, w, r, taxes, hh_parameters);

# Simplified version for one degree of progressivity of labor income and consumption taxes
@elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility_ME(a_grid, 
                                                                    N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters);

# @benchmark compute_hh_taxes_consumption_utility_(a_grid, N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters)

# @elapsed hh_labor_taxes2, hh_consumption2, hh_consumption_tax2, hh_utility2 = compute_hh_taxes_consumption_utility_ME(a_grid, 
#                                                                     N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters);

@benchmark compute_hh_taxes_consumption_utility_ME(a_grid, N_a, rho_grid, l_grid, N_l, w, r, Tau_y, Tau_c, taxes, hh_parameters)


#################### RANDOM CHECK - BUDGET CONSTRAINT HOLDS ###################

test_budget_constraint()

#---------------------# PERFORM VFI - VECTORISED VERSION #--------------------#

@elapsed V_new, policy_a_index, policy_l_index = standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
#@benchmark standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)

@elapsed V_new2, policy_a_index2, policy_l_index2 = MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
@benchmark MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)

############ RANDOM CHECK - BUDGET CONSTRAINT HOLDS FOR SOLUTIONS #############

# Random check: Budget constraint is satisfied for given random household state
# c + T(c) = y - T_y(y) + (1 + r)a - a'

test_optimal_budget_constraint()

# Checks
policy_a = a_grid[policy_a_index[:,:]]
policy_l = a_grid[policy_l_index[:,:]]


############################ PLOT POLICY FUNCTIONS ############################
##Policy function for assets
# Initialize the plot object
plt = plot(title="Policy Function for Assets", 
           xlabel="Current Assets (a)", 
           ylabel="Future Assets (a')", 
           legend=:bottomright)

# Loop over each productivity level and add a line to the plot
for i_rho in 1:N_rho
    plot!(plt, a_grid, policy_a[i_rho, :], label="ρ = $i_rho", lw=2)
end

# Display the plot
display(plt)

## Policy function for labor
# Initialize the plot object
plt = plot(title="Policy Function for Labor", 
           xlabel="Current Assets (a)", 
           ylabel="Labor supplied (l)", 
           legend=:bottomright)

# Loop over each productivity level and add a line to the plot
for i_rho in 1:N_rho
    plot!(plt, a_grid, policy_l[i_rho, :], label="ρ = $i_rho", lw=2)
end

# Display the plot
display(plt)

# Store the VFI guess 
SaveMatrix(V_new, "output/preliminary/V_guess_matrix.txt")
V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")

################ INTERPOLATE ##################

fine_grid = range(a_min, a_max, length=500)  # Finer grid with 500 points

# Store interpolated values for plotting
interp_policy_a = zeros(size(policy_a, 1), length(fine_grid))
a_grid_r = range(a_min, a_max, N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = policy_a[rho_i, :]

    # Create the cubic spline interpolant
    itp = cubic_spline_interpolation(a_grid_r, policy_values, extrapolation_bc=Flat())

    # Evaluate the interpolant on the fine grid
    interp_policy_a[rho_i, :] = itp.(fine_grid)
end

# Plot
# Define colors for each productivity level
colors = palette(:viridis, size(policy_a, 1));

# Plot the interpolated policy functions
p = plot(
    fine_grid, interp_policy_a[1, :],
    label="ρ = $(rho_grid[1])", color=colors[1], linewidth=2,
    xlabel = "Assets (a)", ylabel = "Next Period Assets (a')",
    title = "Smoothed Policy Functions for Assets",
    legend = :bottomright);

for rho_i in 2:N_rho
    plot!(p, fine_grid, interp_policy_a[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Add labels and legend

display(p)
