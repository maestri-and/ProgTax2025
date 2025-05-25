###############################################################################
################################# ROBCHECK.JL #################################

################## This script performs robustness checks on ##################
###################### the benchmark ProgTax(2025) model ######################

###############################################################################

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 0. IMPORTING LIBRARIES AND SUBMODULES #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

using LinearAlgebra
using Distances
using Base.Threads
using Interpolations
using DelimitedFiles
using Plots
using BenchmarkTools
using Dates
using Infiltrator


include("Parameters.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("HouseholdsFirmsGov.jl")
include("Interpolants.jl")
include("SolvingFunctions.jl")
include("PlottingFunctions.jl")

time_start = now()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------# 1. IMPORT BASELINE MODEL RESULTS  #--------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Get steady state values for multiple tax structs
baseline_path = "output/equilibria/baseline/model_results"
baseline = get_model_results(baseline_path)

baseline.taxes = Taxes.(baseline.taxes)

# Extract baseline taxes and government expenditure
b_taxes = baseline.taxes[1]
b_aggG = baseline.aggG[1]
@info("Looking for other regimes generating revenue = $b_aggG")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 2. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

@info("Making grids...")

# Define grid parameters
gpar = GridParams(a_min, 300.000, 400, # Assets
                    0.0, 1, 150,    # Labor
                    length(rho_grid) # Productivity 
                    )

# Assets
a_gtype = "polynomial"
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = a_gtype, pol_power = 4)

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l; grid_type = "labor-double")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------# 3. CHECK 1: FIX CONSUMPTION TAX REVENUE  #----------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

s_c_target = baseline.aggT_c[1] / baseline.aggG[1]
G_target = baseline.aggG[1]
base_taxes = baseline.taxes[1]

# # Look for alternative equilibria at the extrema of the tau_y distribution
# # tau_y - 20% = 0.15, \tau_y + 20% = 0.225
# taxes_r1 = deepcopy(base_taxes)
# taxes_r1.tau_y = 0.15

# # Compute new equilibrium

# new_lambda_c, new_tau_c, r, w, stat_dist, 
# valuef, policy_a, policy_l, policy_c = MultiDimEquilibriumNewton(
#                                            a_grid, rho_grid, l_grid,
#                                             gpar, hhpar, fpar, taxes_r1,
#                                             pi_rho, comp_params, G_target,
#                                             s_c_target;
#                                             target == "revenue",
#                                             initial_x = [0.03, taxes_r1.tau_c, taxes_r1.lambda_c],
#                                             tol = 1e-6,
#                                             max_iter = 50,
#                                             damping_weight = 1
#                                         )
 
# # Adjust taxes
# taxes_r1.lambda_c = new_lambda_c
# taxes_r1.tau_c = new_tau_c

# # Compute other useful distributions and aggregates
# distC, distK, distH, distL,
# distCtax, distWtax, distKtax, 
# aggC, aggK, aggH, aggL, aggG, aggY,
# aggT_c, aggT_y, aggT_k, 
# excess_prod, bc_max_discrepancy = compute_aggregates_and_check(stat_dist, policy_a, policy_c, 
#                                                                     policy_l, rho_grid, a_grid, w, r, taxes_r1;
#                                                                     raise_bc_error = false, raise_clearing_error = false);        


# # Save to file equilibrium details 
# items = Dict(
#     # Calibration details
#     :hhpar => hhpar, :rhopar => rhopar, :fpar => fpar, :taxes => taxes_r1, :gpar => gpar, 
#     # Equilibrium 
#     :r => r, :w => w, :stat_dist => stat_dist,
#     # Policy rules
#     :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c, :valuef => valuef,
#     # Main distributions
#     :distC => distC, :distK => distK, :distH => distH, :distL => distL,
#     :distCtax => distCtax, :distWtax => distWtax, :distKtax => distKtax,
#     # Main aggregates
#     :aggC => aggC, :aggK => aggK, :aggH => aggH, :aggL => aggL, :aggG => aggG, :aggY => aggY,
#     :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
#     # Accuracy stats
#     :excess_prod => excess_prod, :bc_max_discrepancy => bc_max_discrepancy[1],
# )

# for (name, mat) in items
#     filepath = "./output/equilibria/rob_checks/equal_ctax_rev/model_results/" * string(name) * ".txt"
#     SaveMatrix(mat, filepath; overwrite=true, taxes = taxes_r1)
# end

# # Store results as _r1
# for (name, value) in items
#     @eval $(Symbol(string(name), "_r1")) = deepcopy($value)
# end


# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# ------------------# 3. CHECK 2: FIX AVERAGE EFFECTIVE RATE  #----------------#
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# # Extract target average effective rate for consumption
# # Computing consumption tax per each state
# base_taxes = Taxes(baseline.taxes[1])

# consumption_tax_policy = baseline.policy_c[1] .- base_taxes.lambda_c .* baseline.policy_c[1] .^ (1 - base_taxes.tau_c) 
# cons_tax_eff_rates = consumption_tax_policy ./ baseline.policy_c[1]

# aer_target = sum(cons_tax_eff_rates .* baseline.stat_dist[1])
# G_target = baseline.aggG[1]

# # Look for alternative equilibria at the extrema of the tau_y distribution
# # tau_y - 20% = 0.15, \tau_y + 20% = 0.225
# taxes_r2 = deepcopy(base_taxes)
# taxes_r2.tau_y = 0.15

# # Compute new equilibrium

# new_lambda_c, new_tau_c, r, w, stat_dist, 
# valuef, policy_a, policy_l, policy_c = MultiDimEquilibriumNewton(
#                                            a_grid, rho_grid, l_grid,
#                                             gpar, hhpar, fpar, taxes_r2,
#                                             pi_rho, comp_params, G_target,
#                                             aer_target;
#                                             target = "aer",
#                                             # initial_x = [0.03, base_taxes.tau_c, base_taxes.lambda_c],
#                                             initial_x = [0.03, 0.00, 1 - aer_target],
#                                             tol = 1e-6,
#                                             max_iter = 50,
#                                             damping_weight = 1
#                                         )

# # Export results

# # Save to file equilibrium details 
# items = Dict(
#     # Calibration details
#     :hhpar => hhpar, :rhopar => rhopar, :fpar => fpar, :taxes => taxes_r2, :gpar => gpar, 
#     # Equilibrium 
#     :r => r, :w => w, :stat_dist => stat_dist,
#     # Policy rules
#     :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c, :valuef => valuef,
#     # Main distributions
#     :distC => distC, :distK => distK, :distH => distH, :distL => distL,
#     :distCtax => distCtax, :distWtax => distWtax, :distKtax => distKtax,
#     # Main aggregates
#     :aggC => aggC, :aggK => aggK, :aggH => aggH, :aggL => aggL, :aggG => aggG, :aggY => aggY,
#     :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
#     # Accuracy stats
#     :excess_prod => excess_prod, :bc_max_discrepancy => bc_max_discrepancy[1],
# )

# for (name, mat) in items
#     filepath = "./output/equilibria/rob_checks/equal_ctax_aer/model_results/" * string(name) * ".txt"
#     SaveMatrix(mat, filepath; overwrite=true, taxes = taxes_r2)
# end

# # Store results as _r2
# for (name, value) in items
#     @eval $(Symbol(string(name), "_r2")) = deepcopy($value)
# end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------# 4. CHECK 3: FIX LAMBDA_C (TARGET G)  #-----------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# # Adjust lambda_c to make new equilibrium revenue neutral
# base_taxes = baseline.taxes[1]
# G_target = baseline.aggG[1]

# # Look for alternative equilibria at the extrema of the tau_y distribution
# # tau_y - 20% = 0.15
# taxes_r2 = deepcopy(base_taxes)
# taxes_r2.tau_y = 0.15

# # Compute new equilibrium

# # Compute equilibrium
# new_lambda_c, r, w, stat_dist, valuef, policy_a, policy_l, policy_c = TwoLevelEquilibriumNewton(
#             a_grid, rho_grid, l_grid,
#             gpar, hhpar, fpar, taxes_r2,
#             pi_rho, comp_params, b_aggG;
#             adjust_par = :lambda_c
#         )

# # Fix new taxes parameter
# taxes_r2.lambda_c = new_lambda_c

# # Export results

# # Save to file equilibrium details 
# items = Dict(
#     # Calibration details
#     :hhpar => hhpar, :rhopar => rhopar, :fpar => fpar, :taxes => taxes_r2, :gpar => gpar, 
#     # Equilibrium 
#     :r => r, :w => w, :stat_dist => stat_dist,
#     # Policy rules
#     :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c, :valuef => valuef,
#     # Main distributions
#     :distC => distC, :distK => distK, :distH => distH, :distL => distL,
#     :distCtax => distCtax, :distWtax => distWtax, :distKtax => distKtax,
#     # Main aggregates
#     :aggC => aggC, :aggK => aggK, :aggH => aggH, :aggL => aggL, :aggG => aggG, :aggY => aggY,
#     :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
#     # Accuracy stats
#     :excess_prod => excess_prod, :bc_max_discrepancy => bc_max_discrepancy[1],
# )

# for (name, mat) in items
#     filepath = "./output/equilibria/rob_checks/equal_ctax_aer/model_results/" * string(name) * ".txt"
#     SaveMatrix(mat, filepath; overwrite=true, taxes = taxes_r2)
# end

# # Store results as _r2
# for (name, value) in items
#     @eval $(Symbol(string(name), "_r2")) = deepcopy($value)
# end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 3. ROBUSTNESS CHECK ANALYSIS  #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----#----#----#----#----# Re-import data if needed #----#----#----#----#----#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# R1: Revenue share equivalent equilibrium 
r1_path = "output/equilibria/rob_checks/equal_ctax_rev/model_results"
r1 = get_model_results(r1_path)

# Extract values
for col in names(r1)
    val = r1[1, col]
    @eval $(Symbol(string(col), "_r1")) = $val
end

# R2: Average Effective Rate equivalent equilibrium 
# r2_path = "output/equilibria/rob_checks/equal_ctax_aer/model_results"
# r2 = get_model_results(r2_path)

# # Extract values
# for col in names(r2)
#     val = r2[1, col]
#     @eval $(Symbol(string(col), "_r2")) = $val
# end

# Baseline
baseline_path = "output/equilibria/baseline/model_results"
baseline = get_model_results(baseline_path)

# Ensure data type
matrix_cols = ["distC", "distCtax", "distH", "distK", "distKtax", "distL", "distWtax",
               "policy_a", "policy_c", "policy_l", "stat_dist"]
for col in matrix_cols
    baseline[!, col] = Matrix.(baseline[!, col])
end

baseline.taxes = Taxes.(baseline.taxes)

# Extract values
for col in names(baseline)
    val = baseline[1, col]
    @eval $(Symbol(string(col), "_b")) = $val
end

# Alternative regimes results 
keep_baseline = true

# Get steady state values for multiple tax structs
folderpath = "output/equilibria/equivalent_regimes"
# Retrieve model results folder
dirs = filter(isdir, readdir("output/equilibria/equivalent_regimes", join=true))
dirs = joinpath.(dirs, "model_results")

# Extract data and append to baseline
ss = deepcopy(baseline)[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]

for i in eachindex(dirs)
    temp = get_model_results(dirs[i])
    append!(ss, temp)
end

# Ensure data type
matrix_cols = ["distC", "distCtax", "distH", "distK", "distKtax", "distL", "distWtax",
               "policy_a", "policy_c", "policy_l", "stat_dist"]
for col in matrix_cols
    ss[!, col] = Matrix.(ss[!, col])
end

ss.taxes = Taxes.(ss.taxes)

# Remove baseline if needed 
if !keep_baseline
    ss = ss[2:end, :]
end

# Reorder
sort!(ss, :tau_y)

# Extract Augmented Consumption Tax Progressivity Equilibrium data 
# Extract values
for col in names(ss)
    val = ss[1, col]
    @eval $(Symbol(string(col), "_acp")) = $val
end

taxes_acp = Taxes(lambda_y_acp, tau_y_acp, lambda_c_acp, tau_c_acp, tau_k_acp)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----#----#----#----# Compare rob_check with results #----#----#----#----#----#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Create Table for Comparisons using DataFrames
# Append DataFrames
df = deepcopy(baseline)
append!(df, ss[1])
append!(df, r2)
append!(df, r1)

# Pivot with aliases 
df.rowname = ["b", "ss", "r2", "r1"]  # add row identifiers
long = stack(df, Not(:rowname))  # melt into long format
wide = unstack(long, :rowname, :variable, :value)  # pivot rowname into columns

# Divide into ad-hoc sections
