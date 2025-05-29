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
using PrettyTables


include("Parameters.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("HouseholdsFirmsGov.jl")
include("Interpolants.jl")
include("SolvingFunctions.jl")
include("PlottingFunctions.jl")
include("AnalysisFunctions.jl")


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

# s_c_target = baseline.aggT_c[1] / baseline.aggG[1]
# G_target = baseline.aggG[1]
# base_taxes = baseline.taxes[1]

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

# # Compute other useful distributions and aggregates
# distC, distK, distH, distL,
# distCtax, distWtax, distKtax, 
# aggC, aggK, aggH, aggL, aggG, aggY,
# aggT_c, aggT_y, aggT_k, 
# excess_prod, bc_max_discrepancy = compute_aggregates_and_check(stat_dist, policy_a, policy_c, 
#                                                                     policy_l, rho_grid, a_grid, w, r, taxes_r2;
#                                                                     raise_bc_error = false, raise_clearing_error = false);        


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
#     filepath = "./output/equilibria/rob_checks/equal_ctax_tauc/model_results/" * string(name) * ".txt"
#     SaveMatrix(mat, filepath; overwrite=true, taxes = taxes_r2)
# end

# # Store results as _r2
# for (name, value) in items
#     @eval $(Symbol(string(name), "_r2")) = deepcopy($value)
# end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------# 3. PREPARE ANALYSIS  #--------------------------#
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

r1 = r1[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]

# R2: Average Effective Rate equivalent equilibrium 
r2_path = "output/equilibria/rob_checks/equal_ctax_tauc/model_results"
r2 = get_model_results(r2_path)

# Extract values
for col in names(r2)
    val = r2[1, col]
    @eval $(Symbol(string(col), "_r2")) = $val
end

r2 = r2[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]


# Baseline
baseline_path = "output/equilibria/baseline/model_results"
baseline = get_model_results(baseline_path)

# Ensure data type
matrix_cols = ["distC", "distCtax", "distH", "distK", "distKtax", "distL", "distWtax",
               "policy_a", "policy_c", "policy_l", "stat_dist"]
for col in matrix_cols
    baseline[!, col] = Matrix.(baseline[!, col])
end

# Extract values
for col in names(baseline)
    val = baseline[1, col]
    @eval $(Symbol(string(col), "_b")) = $val
end

baseline = baseline[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]

# Alternative regimes results 
keep_baseline = true

# Get steady state values for multiple tax structs
folderpath = "output/equilibria/equivalent_regimes"
# Retrieve model results folder
dirs = filter(isdir, readdir("output/equilibria/equivalent_regimes", join=true))
dirs = joinpath.(dirs, "model_results")

# Extract data and append to baseline
ss = deepcopy(baseline)

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
push!(df, ss[1, :])
append!(df, r2)
append!(df, r1)

df.taxes = Taxes.(zip(df.lambda_y, df.tau_y, df.lambda_c, df.tau_c, df.tau_k))

# Add row identifiers
insertcols!(df, 1, :sim => ["b", "acp", "lc_eq", "Crev_eq"])

# Extract Row Indices
b_idx = findfirst(==("b"), df.sim)
acp_idx = findfirst(==("acp"), df.sim)
lc_eq_idx = findfirst(==("lc_eq"), df.sim)
Crev_eq_idx = findfirst(==("Crev_eq"), df.sim)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------------# ADDING FURTHER INDICATORS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Compute break-even points 
df.bky = df.lambda_y .^ (1 ./ df.tau_y)
df.bkc = ifelse.(df.tau_c .> 0, df.lambda_c .^ (1 ./ df.tau_c), NaN)

#---------------------------------# 1. INCOME #--------------------------------#

# Compute Labor Income and Labor Income Tax Stats
# Preallocate new columns
df.gini_income = similar(df.lambda_y)
df.gini_disp_income = similar(df.lambda_y)
df.policy_l_incpt = similar(df.stat_dist)
df.labor_tax_policy = similar(df.stat_dist)
df.labor_tax_rate = similar(df.stat_dist)
df.income_dist_stats = Vector{Any}(undef, nrow(df))
df.avg_rates_Wtax = Vector{Any}(undef, nrow(df))
df.t10tob50_ptinc_ratio = similar(df.lambda_y)
df.t10tob90_ptinc_ratio = similar(df.lambda_y)
df.t10tob50_atinc_ratio = similar(df.lambda_y)
df.t10tob90_atinc_ratio = similar(df.lambda_y)
df.aetr_Wtax = similar(df.lambda_y)
df.b50t10aetr_Wtax = Vector{Any}(undef, nrow(df))

for i in 1:nrow(df)
    distL = df.distL[i]
    policy_l = df.policy_l[i]
    w = df.w[i]
    stat_dist = Matrix(df.stat_dist[i])

    # Tax parameters from DataFrame columns
    lambda_y = df.lambda_y[i]
    tau_y = df.tau_y[i]

    # === Grodf labor income ===
    policy_l_incpt = policy_l .* rho_grid .* w
    df.policy_l_incpt[i] = policy_l_incpt

    labor_tax_policy = policy_l_incpt .- lambda_y .* policy_l_incpt .^ (1 - tau_y)
    labor_tax_rate = ifelse.((labor_tax_policy .== 0) .&& (policy_l_incpt .== 0),
                              0.0,
                              labor_tax_policy ./ policy_l_incpt)

    df.labor_tax_policy[i] = labor_tax_policy
    df.labor_tax_rate[i] = labor_tax_rate

     # === Gini coefficient ===
    df.gini_income[i] = compute_gini(policy_l_incpt, stat_dist, plot_curve = false)
    df.gini_disp_income[i] = compute_gini(policy_l_incpt .- labor_tax_policy, stat_dist, plot_curve = false)

    # === Income distribution stats ===
    df.income_dist_stats[i] = compute_income_distribution_stats(stat_dist, policy_l_incpt)

    avg_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt; cutoffs = [0.5, 0.9, -0.1])
    df.t10tob50_ptinc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[1][2]
    df.t10tob90_ptinc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[2][2]

    avg_disp_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt .- labor_tax_policy; cutoffs = [0.5, 0.9, -0.1])
    df.t10tob50_atinc_ratio[i] = avg_disp_income_stats[3][2] / avg_disp_income_stats[1][2]
    df.t10tob90_atinc_ratio[i] = avg_disp_income_stats[3][2] / avg_disp_income_stats[2][2]

    # === Average effective rates and AETR ===
    df.avg_rates_Wtax[i] = compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    df.aetr_Wtax[i] = sum(labor_tax_rate .* stat_dist)
    df.b50t10aetr_Wtax[i] = round.((df.avg_rates_Wtax[i][2][2], df.avg_rates_Wtax[i][4][2]), digits = 3)
end

#------------------------------# 2. CONSUMPTION #-----------------------------#
# Preallocate columns
df.consumption_tax_policy = similar(df.stat_dist)
df.consumption_tax_rate = similar(df.stat_dist)
df.consumption_tax_rate_vat = similar(df.stat_dist)
df.consumption_tax_gini = similar(df.lambda_c)
df.consumption_base_gini = similar(df.lambda_c)
df.consumption_gini = similar(df.lambda_c)
df.kakwani_cons_tax = similar(df.lambda_c)
df.cons_dist_stats = Vector{Any}(undef, nrow(df))
df.t10tob50_cons_ratio = similar(df.lambda_c)
df.t10tob90_cons_ratio = similar(df.lambda_c)
df.avg_rates_Ctax = Vector{Any}(undef, nrow(df))
df.avg_rates_Ctax_vat = Vector{Any}(undef, nrow(df))
df.b10t10_rates_Ctax = Vector{Any}(undef, nrow(df))
df.cons_deciles = Vector{Vector{Float64}}(undef, nrow(df))

for i in 1:nrow(df)
    policy_c = df.policy_c[i]
    stat_dist = Matrix(df.stat_dist[i])

    lambda_c = df.lambda_c[i]
    tau_c = df.tau_c[i]

    # Compute consumption distribution deciles
    df.cons_deciles[i] = compute_decile_distribution(df.stat_dist[i], df.policy_c[i])

    # Compute consumption taxes paid
    consumption_tax_policy = policy_c .- lambda_c .* policy_c .^ (1 - tau_c)
    df.consumption_tax_policy[i] = consumption_tax_policy
    consumption_plus_tax_policy = policy_c .+ consumption_tax_policy
    df.consumption_tax_rate[i] = consumption_tax_policy ./ consumption_plus_tax_policy
    df.consumption_tax_rate_vat[i] = consumption_tax_policy ./ policy_c
    
    # Gini coefficients
    df.consumption_tax_gini[i] = compute_gini(consumption_tax_policy, stat_dist; plot_curve = false)
    df.consumption_base_gini[i] = compute_gini(consumption_plus_tax_policy, stat_dist; plot_curve = false)
    df.consumption_gini[i] = compute_gini(policy_c, stat_dist; plot_curve = false)

    # === Consumption distribution stats ===    
    df.cons_dist_stats[i] = compute_income_distribution_stats(stat_dist, policy_c)

    avg_cons_stats = compute_average_income_stats(stat_dist, policy_c; cutoffs = [0.5, 0.9, -0.1])

    df.t10tob50_cons_ratio[i] = avg_cons_stats[3][2] / avg_cons_stats[1][2]
    df.t10tob90_cons_ratio[i] = avg_cons_stats[3][2] / avg_cons_stats[2][2]


    # Kakwani Index
    df.kakwani_cons_tax[i] = df.consumption_tax_gini[i] - df.consumption_base_gini[i]

    # === Average effective rates and AETR ===
    df.avg_rates_Ctax[i] = compute_average_rate_stats(stat_dist, consumption_tax_policy, policy_l_incpt = consumption_plus_tax_policy,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    df.avg_rates_Ctax_vat[i] = compute_average_rate_stats(stat_dist, consumption_tax_policy, policy_l_incpt = policy_c,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    df.b10t10_rates_Ctax[i] = round.((df.avg_rates_Ctax_vat[i][1][2], df.avg_rates_Ctax_vat[i][4][2]), digits = 3)

    
end

# compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
#                            cutoffs = [0.1, 0.5, 0.9, -0.1])

# Get extrema for consumption tax rates
df.cons_tax_rates_min = [round.(t; digits=3) for t in minimum.(df.consumption_tax_rate_vat)]
df.cons_tax_rates_max = [round.(t; digits=3) for t in maximum.(df.consumption_tax_rate_vat)]
df.cons_tax_rates_extrema = [round.(t; digits=3) for t in extrema.(df.consumption_tax_rate_vat)]


# Compute % variation from the baseline for each column
df_pct = DataFrame()
float_vars = ["lambda_y", "tau_y", "lambda_c", "tau_c", "tau_k", 
              "aggC", "aggG", "aggH", "aggK", "aggL", "aggT_c", 
              "aggT_k", "aggT_y", "aggY", "r", "w"]

for col in float_vars
    baseline_val = baseline[1, col]
    df_pct[!, col] = 100 .* (df[!, col] .- baseline_val) ./ baseline_val
end

# Add % Variation for progredfivity parameters also to main dataframe
for col in ["tau_y", "tau_c"]
    baseline_val = baseline[1, col]
    new_col = "delta_" * col
    df[!, new_col] = 100 .* (df[!, col] .- baseline_val) ./ baseline_val
end 

# Add Revenue shares 
df.TRevC = df.aggT_c ./ df.aggG
df.TRevW = df.aggT_y ./ df.aggG
df.TRevK = df.aggT_k ./ df.aggG

#----#----#----# Welfare #----#----#----#

# Compute cev and aggCEV for all steady states
df.cev = [zeros(gpar.N_rho, gpar.N_a) for _ in 1:nrow(df)]
df.aggCEV = zeros(Float64, nrow(df))
df.cev_deciles = [zeros(Float64, 10) for _ in 1:nrow(df)] 
df.cev_by_inc_decile = [zeros(Float64, 10) for _ in 1:nrow(df)] 

for i in setdiff(1:nrow(df), b_idx)
    df.cev[i], df.aggCEV[i] = compute_cev(df.valuef[i], df.valuef[b_idx], df.policy_c[b_idx], 
            df.policy_a[b_idx], df.stat_dist[b_idx], pi_rho)
    
    df.cev_deciles[i] = compute_decile_distribution(df.stat_dist[i], df.cev[i])
    
    df.cev_by_inc_decile[i] = compute_avg_cevs_per_income_decile(df.cev[i], df.policy_l_incpt[b_idx], df.stat_dist[i]) ./ 10
end

# Compute Gini for wealth distribution 
df.wealth_gini = similar(df.lambda_c)
df.asset_deciles = similar(df.cons_deciles)
df.hours_deciles = similar(df.cons_deciles)


for i in 1:nrow(df)
    df.wealth_gini[i] = compute_gini(df.policy_a[i], df.stat_dist[i])
    df.asset_deciles[i] = compute_decile_distribution(df.stat_dist[i], df.policy_a[i])
    df.hours_deciles[i] = compute_decile_distribution(df.stat_dist[i], df.policy_l[i])
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----#----#----#----# Pivot and extract table data #----#----#----#----#----#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Pivot with aliases 
tab_df = stack(df, Not(:sim))  # melt into long format
tab_df = unstack(tab_df, :sim, :value)  # pivot rowname into columns

# Generate robustness check results table
rc_table = extract_acp_vs_rob(tab_df, "aggG", table_string = "Government Expenditure", 
                                normalise = "num")
append!(rc_table, extract_acp_vs_rob(tab_df, "lambda_y", table_string = LaTeXString("\\lambda_y")))
append!(rc_table, extract_acp_vs_rob(tab_df, "tau_y", table_string = LaTeXString("\\tau_y")))
append!(rc_table, extract_acp_vs_rob(tab_df, "lambda_c", table_string = LaTeXString("\\lambda_c")))
append!(rc_table, extract_acp_vs_rob(tab_df, "tau_c", table_string = LaTeXString("\\tau_c")))
append!(rc_table, extract_acp_vs_rob(tab_df, "tau_k", table_string = LaTeXString("\\tau_k")))

# Section 1: Taxes details
append!(rc_table, extract_acp_vs_rob(tab_df, "b50t10aetr_Wtax", as_percentage = true, 
                                     table_string = "AETR (%) - Labor Income (Bottom 50%, Top 10%)"))

append!(rc_table, extract_acp_vs_rob(tab_df, "b10t10_rates_Ctax", as_percentage = true, 
                                     table_string = "AETR (%) - Consumption (Bottom 10%, Top 10%)"))

append!(rc_table, extract_acp_vs_rob(tab_df, "cons_tax_rates_extrema", as_percentage = true, 
                                     table_string = "Consumption Tax Effective Rates (%) (Min, Max)"))
###
append!(rc_table, extract_acp_vs_rob(tab_df, "TRevW", as_percentage = true, 
                                     table_string = "Labor Taxes (% of Revenue)"))

append!(rc_table, extract_acp_vs_rob(tab_df, "TRevC", as_percentage = true, 
                                     table_string = "Consumption Taxes (% of Revenue)"))

append!(rc_table, extract_acp_vs_rob(tab_df, "TRevK", as_percentage = true, 
                                     table_string = "Capital Taxes (% of Revenue)"))

# Polish and Output Robustness Check Tax Summary #
# Round
# rc_table.b .= round.(rc_table.b, digits=3)
# rc_table.acp .= round.(rc_table.acp, digits=3)
# rc_table.lc_eq .= round.(rc_table.lc_eq, digits=3)
# rc_table.Crev_eq .= round.(rc_table.Crev_eq, digits=3)
# Header
rename!(rc_table, ["Indicator", "Baseline", "ACP", "LC-EQ", "CRev-EQ"])

open("output/tables/rob_checks/rob_checks_tax_summary.tex", "w") do io
    pretty_table(io, rc_table, backend = Val(:latex))
end

# Plot AER across consumption taxes 
aer_b(c) = (1 - lambda_c_b * c ^ (-tau_c_b)) *100
aer_acp(c) = (1 - lambda_c_acp * c ^ (-tau_c_acp)) *100
aer_alp(c) = (1 - 0.85 * c ^ (0.001)) *100
aer_lc_eq(c) = (1 - lambda_c_r2 * c ^ (-tau_c_r2)) *100
aer_Crev_eq(c) = (1 - lambda_c_r1 * c ^ (-tau_c_r1)) *100

p = plot_function_family([aer_b, aer_acp, aer_alp, aer_lc_eq, aer_Crev_eq], 
    ["Baseline", "ACP", "ALP", "LC-EQ", "CRev-EQ"], 
    minimum(policy_c_b), maximum(policy_c_b);
    laby = "AER (%)",
    labx = "Consumption Expenditure",
    cmap = :Set1_8,
    # y_low = aer_Crev_eq(minimum(policy_c_b)), 
    y_low = -56.0,
    y_up = aer_Crev_eq(maximum(policy_c_b)) + 1,
    size = (500, 500),
    y_ticks = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
)

CairoMakie.save("output/figures/rob_checks/aers_rc.png", p)

# Where do curves cross?
acp_lc_eq_cross(c) = aer_acp(c) - aer_lc_eq(c)
cp_acp_lc_eq = find_zero(acp_lc_eq_cross, 0.2)

acp_Crev_eq_cross(c) = aer_acp(c) - aer_Crev_eq(c)
cp_acp_Crev_eq = find_zero(acp_Crev_eq_cross, 0.2)

# How many households below/above?
idx = sortperm(vec(policy_c_acp))
policy_c_acp_vec = vec(policy_c_acp)[idx]
acp_c_dist = stat_dist_acp[idx]

threshold_lc_eq = findfirst(x -> x > cp_acp_lc_eq, policy_c_acp_vec)
hh_below_lc_eq = sum(acp_c_dist[1:threshold_lc_eq])

threshold_Crev_eq = findfirst(x -> x > cp_acp_Crev_eq, policy_c_acp_vec)
hh_below_Crev_eq = sum(acp_c_dist[1:threshold_Crev_eq])

# Plot AMR across consumption taxes 
amr_b(c) = (1 - lambda_c_b * (1-tau_c_b) * c ^ (-tau_c_b)) *100
amr_acp(c) = (1 - lambda_c_acp * (1-tau_c_acp) * c ^ (-tau_c_acp)) *100
amr_alp(c) = (1 - 0.85 * (1+0.001) * c ^ (+0.001)) *100
amr_lc_eq(c) = (1 - lambda_c_r2 * (1-tau_c_r2) * c ^ (-tau_c_r2)) *100
amr_Crev_eq(c) = (1 - lambda_c_r1 * (1-tau_c_r1) * c ^ (-tau_c_r1)) *100

# amr_b(c) = lambda_c_b * (tau_c_b) * c ^ (-1 -tau_c_b) *100
# amr_acp(c) = lambda_c_acp * (tau_c_acp)* c ^ (-1 -tau_c_acp) *100
# amr_lc_eq(c) = lambda_c_r2 * (tau_c_r2)* c ^ (-1 -tau_c_r2) *100
# amr_Crev_eq(c) = lambda_c_r1 * (tau_c_r1) * c ^ (-1 -tau_c_r1) *100

p = plot_function_family([amr_b, amr_acp, amr_alp, amr_lc_eq, amr_Crev_eq], 
    ["Baseline", "ACP", "ALP", "LC-EQ", "CRev-EQ"], minimum(policy_c_b), maximum(policy_c_b);
    laby = "MR (%)",
    labx = "Consumption Expenditure",
    cmap = :Set1_8,
    y_low = amr_Crev_eq(minimum(policy_c_b)), 
    y_up = amr_Crev_eq(maximum(policy_c_b)) + 1,
    size = (500, 500),
    y_ticks = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    leg_pos = :lt
)


CairoMakie.save("output/figures/rob_checks/amrs_rc.png", p)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 4. ROBUSTNESS CHECK ANALYSIS  #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----#----#----#----#----# Aggregates #----#----#----#----#----#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

sec2_table = extract_acp_vs_rob(tab_df, "aggY", table_string = "Δ Aggregate Output (%)", 
                                normalise = "var", as_percentage = true, return_baseline = false)
append!(sec2_table, extract_acp_vs_rob(tab_df, "aggC", table_string = "Δ Aggregate Consumption (%)", 
                                normalise = "var", as_percentage = true, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "aggK", table_string = "Δ Aggregate Capital (%)", 
                                normalise = "var", as_percentage = true, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "aggH", table_string = "Δ Aggregate Labor (Hours) (%)", 
                                normalise = "var", as_percentage = true, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "w", table_string = "Δ Equilibrium Wage (%)", 
                                normalise = "var", as_percentage = true, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "r", table_string = "Interest Rate (%)", 
                                as_percentage = true, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "gini_income", table_string = "Gini Coefficient for Pre-Tax Income", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "gini_disp_income", table_string = "Gini Coefficient for After-Tax Income", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "consumption_gini", table_string = "Gini Coefficient for Consumption", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "wealth_gini", table_string = "Gini Coefficient for Wealth", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "t10tob50_ptinc_ratio", table_string = "Pre-Tax Income - T10/B50 Average Ratio", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "t10tob50_atinc_ratio", table_string = "After-Tax Income - T10/B50 Average Ratio", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "t10tob50_cons_ratio", table_string = "Consumption - T10/B50 Average Ratio", 
                                normalise = "no", as_percentage = false, return_baseline = false))
append!(sec2_table, extract_acp_vs_rob(tab_df, "aggCEV", table_string = "CEV (%)", 
                                normalise = "no", as_percentage = false, return_baseline = false))


# Output Robustness Check Aggregate Summary #
sec2_table.acp .= round.(sec2_table.acp, digits=3)
sec2_table.lc_eq .= round.(sec2_table.lc_eq, digits=3)
sec2_table.Crev_eq .= round.(sec2_table.Crev_eq, digits=3)
rename!(sec2_table, ["Indicator", "ACP", "LC-EQ", "CRev-EQ"])
open("output/tables/rob_checks/rob_checks_tax_summary.tex", "w") do io
    pretty_table(io, sec2_table, backend = Val(:latex))
end




#----------------# Distributions #----------------#

#----------------# Consumption #----------------#
# Compare differences 
decCdiff = Dict(:acpdiff => df.cons_deciles[acp_idx] - df.cons_deciles[b_idx],
                :lcdiff => df.cons_deciles[lc_eq_idx] - df.cons_deciles[b_idx],
                :crevdiff => df.cons_deciles[Crev_eq_idx] - df.cons_deciles[b_idx])
            
p = plot_decile_distributions_by_group(decCdiff,
    [:acpdiff, :lcdiff, :crevdiff];
    legend_pos = :lb,
    title = "Consumption Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = palette(:Set1_7)[[3,2,1]],
    leg_labels = ["ACP", "LC-EQ", "CRev-EQ"]
    )

CairoMakie.save("output/figures/rob_checks/rc_cons_diff_dist_by_decile.png", p)


#----------------# Capital #----------------#
# Compare differences 
decKdiff = Dict(:acpdiff => df.asset_deciles[acp_idx] - df.asset_deciles[b_idx],
                :lcdiff => df.asset_deciles[lc_eq_idx] - df.asset_deciles[b_idx],
                :crevdiff => df.asset_deciles[Crev_eq_idx] - df.asset_deciles[b_idx])
            
p = plot_decile_distributions_by_group(decKdiff,
    [:acpdiff, :lcdiff, :crevdiff];
    legend_pos = :lt,
    title = "Asset Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = palette(:Set1_7)[[3,2,1]],
    leg_labels = ["ACP", "LC-EQ", "CRev-EQ"]
    )

CairoMakie.save("output/figures/rob_checks/rc_asset_diff_dist_by_decile.png", p)

#----------------# Labor Hours #----------------#
# Compare differences 
decHdiff = Dict(:acpdiff => df.hours_deciles[acp_idx] - df.hours_deciles[b_idx],
                :lcdiff => df.hours_deciles[lc_eq_idx] - df.hours_deciles[b_idx],
                :crevdiff => df.hours_deciles[Crev_eq_idx] - df.hours_deciles[b_idx])
            
p = plot_decile_distributions_by_group(decHdiff,
    [:acpdiff, :lcdiff, :crevdiff];
    legend_pos = :lb,
    title = "Labor (Hours) Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = palette(:Set1_7)[[3,2,1]],
    leg_labels = ["ACP", "LC-EQ", "CRev-EQ"]
    )

CairoMakie.save("output/figures/rob_checks/rc_hours_diff_dist_by_decile.png", p)

#----------------# CEV #----------------#
# Compare differences 
decCEVdiff = Dict(:acpdiff => df.cev_by_inc_decile[acp_idx],
                :lcdiff => df.cev_by_inc_decile[lc_eq_idx],
                :crevdiff => df.cev_by_inc_decile[Crev_eq_idx])
            
p = plot_decile_distributions_by_group(decCEVdiff,
    [:acpdiff, :lcdiff, :crevdiff];
    legend_pos = :lb,
    title = "CEV Distribution by Income Decile - % Difference with Baseline",
    ylabel = "CEV (%)",
    bar_palette = palette(:Set1_7)[[3,2,1]],
    leg_labels = ["ACP", "LC-EQ", "CRev-EQ"],
    as_percentage = false
    )

CairoMakie.save("output/figures/rob_checks/rc_cev_diff_dist_by_decile.png", p)



