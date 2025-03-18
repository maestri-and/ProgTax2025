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
using Dates
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

# Format date for temporary outputs - TBM
ddmm = Dates.format(today(), "mm-dd")


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
# taxes = Taxes(0.7, 0.2, # lambda_y, tau_y, 
#             0.7, 0.136, #lambda_c, tau_c,
#             0.0 # tau_k
#             )

# No taxes - λ=1, τ=0
taxes = Taxes(0.7, 0.2,     # lambda_y, tau_y, 
            0.7, 0.136,       # lambda_c, tau_c,
            0.0             # tau_k
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
hh_labor_taxes, hh_consumption, hh_consumption_tax = compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, taxes)

# cExp2cInt = interp_consumption(hh_consumption, hh_consumption_plus_tax)

#################### RANDOM CHECK - BUDGET CONSTRAINT HOLDS ###################

test_budget_constraint()

#---------# PERFORM VFI - INTERPOLATED VERSION EXPLOITING LABOR FOC #---------#

println("Pinning down optimal labor and consumption using labor FOC...")

# Interpolate functions that define optimal labor supply and 
# optimal consumption (and therefore utility level) for each (ρ, a, a')
# Using labor FOC and budget constraint 

opt_c_FOC, opt_l_FOC = find_opt_cons_labor(rho_grid, a_grid, w, net_r, taxes, hh_parameters, gpar) #TBM - can be optimised

# --- Interpolate Optimal Labor and Consumption as functions of a' for each (ρ, a) ---#
opt_c_itp, opt_l_itp, opt_u_itp, max_a_prime = interp_opt_funs(a_grid, opt_c_FOC, opt_l_FOC, gpar, hh_parameters);

# --- Launch VFI ---#
println("Launching VFI...")

valuef, policy_a = intVFI_FOC_parallel(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hh_parameters, gpar, comp_params)

# Interpolate value function and policy function for assets 
valuef_int = Spline2D(rho_grid, a_grid, valuef)
policy_a_int = Spline2D(rho_grid, a_grid, policy_a)

# Extract policy functions for labor and consumption using FOC-derived interpolations
policy_l = compute_policy_matrix(opt_l_itp, policy_a_int, a_grid, rho_grid)
policy_l_int = Spline2D(rho_grid, a_grid, policy_l)

policy_c = compute_policy_matrix(opt_c_itp, policy_a_int, a_grid, rho_grid)
policy_c_int = Spline2D(rho_grid, a_grid, policy_c)


# --- Plot value and policy functions --- # 
plot_value_function(valuef, a_grid, rho_grid)
savefig("output/preliminary/policy_funs/cont/value_function_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")

plot_policy_function(policy_a_int, a_grid, rho_grid, policy_type = "assets")
savefig("output/preliminary/policy_funs/cont/asset_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")

plot_policy_function(policy_l_int, a_grid, rho_grid, policy_type = "labor")
savefig("output/preliminary/policy_funs/cont/labor_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")

plot_policy_function(policy_c_int, a_grid, rho_grid, policy_type = "consumption")
savefig("output/preliminary/policy_funs/cont/cons_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")

plot_policy_function(policy_c, a_grid, rho_grid, policy_type="assets")

# Adjust grids


#######################################################################################

# Store the VFI guess 
SaveMatrix(V_new, "output/preliminary/V_guess_matrix_a" * "$gpar.N_a" * "_l" * "$gpar.N_l" * ".txt")
# V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")

# Save the figure
# savefig(pfa, "output/preliminary/asset_policy_len$gpar.N_a.png")

# Save the figure
# savefig(pfl, "output/preliminary/labor_policy_l$gpar.N_l" * "_a$gpar.N_a" * ".png")

