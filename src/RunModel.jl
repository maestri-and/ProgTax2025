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
# using Infiltrator


include("Parameters.jl")
include("FirmsGov.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("Households.jl")
include("Interpolants.jl")
include("SolvingFunctions.jl")
include("PlottingFunctions.jl")
include("../tests/TestingFunctions.jl")


println("Starting model solution...")
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

println("Making grids...")

# Assets
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a)

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l)

# Labor productivity - Defined in model_parameters.jl
# rho_grid = rho_grid

# Taxation parameters
taxes = Taxes(0.7, 0.2, # lambda_y, tau_y, 
            0.7, 0.136, #lambda_c, tau_c,
            0.0 # tau_k
            )

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 2. VALUE FUNCTION ITERATION #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Temporary (TBM)
w = 1
r = 0.05
net_r = (1 - taxes.tau_k)r


#### COMPUTING TAXES, CONSUMPTION AND UTILITY FOR EACH STATE-CHOICE POINT #####

println("Solving budget constraint...")

## INTERPOLATE TO SAVE MEMORY ## #TBM - can be optimised
hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_consumption_plus_tax = compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, taxes)

cExp2cInt = interp_consumption(hh_consumption, hh_consumption_plus_tax)

#################### RANDOM CHECK - BUDGET CONSTRAINT HOLDS ###################

test_budget_constraint()

#---------# PERFORM VFI - INTERPOLATED VERSION EXPLOITING LABOR FOC #---------#

println("Computing optimal labor and consumption using labor FOC...")

# Interpolate functions that define optimal labor supply and 
# optimal consumption (and therefore utility level) for each (ρ, a, a')
# Using labor FOC and budget constraint 

opt_c_FOC, opt_l_FOC = find_opt_cons_labor(rho_grid, a_grid, w, net_r, taxes, hh_parameters, gpar) #TBM - can be optimised

# --- Interpolate Optimal Labor and Consumption as functions of a' for each (ρ, a) ---#
opt_c_itp, opt_l_itp, opt_u_itp, max_a_prime = interp_opt_funs(a_grid, opt_c_FOC, opt_l_FOC, gpar, hh_parameters);

# --- Launch VFI ---#
println("Launching VFI...")

V_new, policy_a = intVFI_FOC_parallel(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hh_parameters, gpar, comp_params)

# --- Plot policy function --- # 
plot_policy_function(policy_a2, a_grid, rho_grid, policy_type="assets")



#######################################################################################

# Store the VFI guess 
SaveMatrix(V_new, "output/preliminary/V_guess_matrix_a" * "$gpar.N_a" * "_l" * "$gpar.N_l" * ".txt")
# V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")

# Save the figure
# savefig(pfa, "output/preliminary/asset_policy_len$gpar.N_a.png")

# Save the figure
# savefig(pfl, "output/preliminary/labor_policy_l$gpar.N_l" * "_a$gpar.N_a" * ".png")

################ EX-POST INTERPOLATIONS ##################

# TBM - To be exported to other script

println("Interpolating results...")
###### Interpolate Asset policy function ######

fine_grid_a = range(gpar.a_min, gpar.a_max, length=2*gpar.N_a) 

# Store interpolated values for plotting
interp_policy_a = zeros(size(policy_a, 1), length(fine_grid_a))
a_grid_r = range(gpar.a_min, gpar.a_max, gpar.N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = policy_a[rho_i, :]

    # Create the cubic spline interpolant
    itp = cubic_spline_interpolation(a_grid_r, policy_values, extrapolation_bc=Interpolations.Flat())

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

for rho_i in 2:gpar.N_rho    plot!(pfa_int, fine_grid_a, interp_policy_a[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfa_int)

# savefig(pfa_int, "output/preliminary/asset_policy_int_len$gpar.N_a.png")



###### Interpolate Labor policy function ######

# Store interpolated values for plotting
interp_policy_l = zeros(size(policy_l, 1), length(fine_grid_a))
a_grid_r = range(gpar.a_min, gpar.a_max, gpar.N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = vec(policy_l[rho_i, :])

    # Create the cubic spline interpolant
    itp = linear_interpolation(a_grid_r, policy_values, extrapolation_bc=Interpolations.Flat())

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

for rho_i in 2:gpar.N_rho    plot!(pfl_int, fine_grid_a, interp_policy_l[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfl_int)

# Create filename and save
# filename = "asset_policy_int_l" * "$gpar.N_l" * "_a" * "$gpar.N_a" * ".png"
# savefig(pfl_int, "output/preliminary/" * filename)

