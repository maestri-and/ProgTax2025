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


include("Parameters.jl")
include("FirmsGov.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("Households.jl")
include("SolvingFunctions.jl")
include("../tests/TestingFunctions.jl")


println("Starting model solution...")
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

println("Making grids...")

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

println("Solving budget constraint...")

## INTERPOLATE TO SAVE MEMORY ##
@elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_consumption_plus_tax = compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, N_a, N_rho, N_l, w, r, Tau_y, Tau_c, taxes)

cExp2cInt = interp_consumption(hh_consumption, hh_consumption_plus_tax)

#################### RANDOM CHECK - BUDGET CONSTRAINT HOLDS ###################

test_budget_constraint()

#---------------------# PERFORM VFI - VECTORISED VERSION #--------------------#

println("Launching VFI...")

# @elapsed V_new, policy_a_index, policy_l_index = standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
#@benchmark standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)

@elapsed V_new, policy_a_index, policy_l_index = MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho;
                        # V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")
                        )
# @benchmark MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)

############ RANDOM CHECK - BUDGET CONSTRAINT HOLDS FOR SOLUTIONS #############

# Random check: Budget constraint is satisfied for given random household state
# c + T(c) = y - T_y(y) + (1 + r)a - a'

test_optimal_budget_constraint()


# Store the VFI guess 
SaveMatrix(V_new, "output/preliminary/V_guess_matrix_a" * "$N_a" * "_l" * "$N_l" * ".txt")
# V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")

# Checks
policy_a = a_grid[policy_a_index[:,:]]
policy_l = l_grid[policy_l_index[:,:]]


############################ PLOT POLICY FUNCTIONS ############################
##Policy function for assets
# Initialize the plot object
pfa = plot(title="Policy Function for Assets", 
           xlabel="Current Assets (a)", 
           ylabel="Future Assets (a')", 
           legend=:bottomright);

# Loop over each productivity level and add a line to the plot
for i_rho in 1:N_rho
    plot!(pfa, a_grid, policy_a[i_rho, :], label="ρ = $i_rho", lw=2)
end

# Display the plot
display(pfa)

# Save the figure
# savefig(pfa, "output/preliminary/asset_policy_len$N_a.png")


## Policy function for labor
# Initialize the plot object
pfl = plot(title="Policy Function for Labor", 
           xlabel="Current Assets (a)", 
           ylabel="Labor supplied (l)", 
           legend=:bottomright)

# Loop over each productivity level and add a line to the plot
for i_rho in 1:N_rho
    plot!(pfl, a_grid, policy_l[i_rho, :], label="ρ = $i_rho", lw=2)
end

# Display the plot
display(pfl)

# Save the figure
# savefig(pfl, "output/preliminary/labor_policy_l$N_l" * "_a$N_a" * ".png")

################ INTERPOLATE ##################

# TBM - To be exported to other script

println("Interpolating results...")
###### Interpolate Asset policy function ######

fine_grid_a = range(a_min, a_max, length=2*N_a) 

# Store interpolated values for plotting
interp_policy_a = zeros(size(policy_a, 1), length(fine_grid_a))
a_grid_r = range(a_min, a_max, N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = policy_a[rho_i, :]

    # Create the cubic spline interpolant
    itp = cubic_spline_interpolation(a_grid_r, policy_values, extrapolation_bc=Flat())

    # Evaluate the interpolant on the fine grid
    interp_policy_a[rho_i, :] = itp.(fine_grid_a)
end

# Plot
# Define colors for each productivity level
colors = palette(:viridis, size(policy_a, 1));

# Plot the interpolated policy functions
pfa_int = plot(
    fine_grid_a, interp_policy_a[1, :],
    label="ρ = $(rho_grid[1])", color=colors[1], linewidth=2,
    xlabel = "Assets (a)", ylabel = "Next Period Assets (a')",
    title = "Policy Functions for Assets",
    legend = :bottomright);

for rho_i in 2:N_rho
    plot!(pfa_int, fine_grid_a, interp_policy_a[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfa_int)

# savefig(pfa_int, "output/preliminary/asset_policy_int_len$N_a.png")



###### Interpolate Labor policy function ######

# Store interpolated values for plotting
interp_policy_l = zeros(size(policy_l, 1), length(fine_grid_a))
a_grid_r = range(a_min, a_max, N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = vec(policy_l[rho_i, :])

    # Create the cubic spline interpolant
    itp = linear_interpolation(a_grid_r, policy_values, extrapolation_bc=Flat())

    # Evaluate the interpolant on the fine grid
    interp_policy_l[rho_i, :] = itp.(fine_grid_a)
end

# Plot
# Define colors for each productivity level
colors = palette(:viridis, size(policy_l, 1));

# Plot the interpolated policy functions
pfl_int = plot(
    fine_grid_a, interp_policy_l[1, :],
    label="ρ = $(rho_grid[1])", color=colors[1], linewidth=2,
    xlabel = "Assets (a)", ylabel = "Labor choice (ℓ)",
    title = "Policy Functions for Labor",
    legend = :bottomright);

for rho_i in 2:N_rho
    plot!(pfl_int, fine_grid_a, interp_policy_l[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfl_int)

# Create filename and save
# filename = "asset_policy_int_l" * "$N_l" * "_a" * "$N_a" * ".png"
# savefig(pfl_int, "output/preliminary/" * filename)

