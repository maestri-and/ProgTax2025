###############################################################################
############################# RUNBASELINEMODEL.JL #############################

####################### This script solves and analyses #######################
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
# using DataFrames
using Plots
using BenchmarkTools
using Dates
using Infiltrator
using QuantEcon
# using CairoMakie: surface, Figure, Axis3


include("Parameters.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("HouseholdsFirmsGov.jl")
include("Interpolants.jl")
include("SolvingFunctions.jl")
include("PlottingFunctions.jl")
include("../tests/TestingFunctions.jl")
include("AnalysisFunctions.jl")

# Format date for temporary outputs
ddmm = Dates.format(today(), "mm-dd")
time_start = now()
timestamp_start = Dates.format(now(), "yyyymmdd-HH_MM_SS")

@info("Starting model solution...")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
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

# Labor productivity - Defined in Parameters.jl
# Extract stable distribution from transition matrix
rho_dist = find_stable_dist(pi_rho)

# Taxation parameters - baseline calibration
taxes = Taxes(0.7305, 0.1875, # lambda_y, tau_y, 
0.85, 0.0343, #lambda_c, tau_c,
0.352 # tau_k
)

# Taxation parameters - no taxes            
# taxes = Taxes(1.0, 0.0, # lambda_y, tau_y, 
# 1.0, 0.0, #lambda_c, tau_c,
# 0.0 # tau_k
# )

# # Taxation parameters - Custom taxes            
# taxes = Taxes(0.7, 0.475, # lambda_y, tau_y, 
# 0.7, 0.475, #lambda_c, tau_c,
# 0.3 # tau_k
# )

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------------# 2. SOLVING MODEL #-----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

@info("Solving for λ_y: $(taxes.lambda_y),  τ_y: $(taxes.tau_y), λ_c: $(taxes.lambda_c), τ_c: $(taxes.tau_c), τ_k: $(taxes.tau_k)")

# Compute equilibrium
r, w, stat_dist, valuef, policy_a, policy_l, policy_c, 
rates, errors = ComputeEquilibrium_Newton(a_grid, rho_grid, l_grid, 
                                    gpar, hhpar, fpar, taxes,
                                    pi_rho, comp_params; 
                                    prevent_Newton_jump = false,
                                    initial_r = 0.03)

# Compute other useful distributions and aggregates
distC, distK, distH, distL,
distCtax, distWtax, distKtax, 
aggC, aggK, aggH, aggL, aggG, aggY,
aggT_c, aggT_y, aggT_k, 
excess_prod, bc_max_discrepancy = compute_aggregates_and_check(stat_dist, policy_a, policy_c, 
                                                                    policy_l, rho_grid, a_grid, w, r, taxes;
                                                                    raise_bc_error = false, raise_clearing_error = false);        


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------------# 3. PLOTS AND ANALYSIS #--------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#-----------------------# 1. VALUE FUNCTION ITERATION #-----------------------#


# Plot rates vs errors
# Plots.scatter(rates, errors, xlabel = "Interest rate", ylabel = "Capital market error")

# Interpolate and return value function and policy functions
# valuef_int, policy_a_int, policy_c_int, policy_l_int = interpolate_policy_funs(valuef, policy_a, policy_c, policy_l, rho_grid, a_grid);

# Plot policy functions if necessary
plot_household_policies(valuef, policy_a, policy_l, policy_c,
                                 a_grid, rho_grid, taxes;
                                 plot_types = ["value", "assets", "labor", "consumption"],
                                 save_plots = false)

# 3D plot: labor policy function
# plot_policy_function_3d(policy_l, a_grid, rho_grid; policy_type="labor")
# plot_policy_function(policy_l, a_grid, rho_grid; policy_type="labor")


# Plot stationary distribution 
# plot_heatmap_stationary_distribution(stat_dist; taxes=taxes)
# plot_density_by_productivity(stat_dist, a_grid, gpar; rho_grid=nothing)


#---------------------------------# 2. WEALTH #--------------------------------#

# Compute capital-to-output ratio
KtoY = aggK / aggY

# Extract main distributions - wealth distribution
distA_stats = compute_wealth_distribution_stats(stat_dist, a_grid; 
                                                cutoffs = [0.5, 0.6, 0.8, -0.1, -0.05, -0.01], 
                                                replace_debt = false)

plot_dist_stats_bar(distA_stats)

# Plot vs data 
# Import wealth data 
wealth_data, header = readdlm("data/wealth/WID_wealth_stats-2008-2023.csv", ',', header = true)
wealth_df = DataFrame(wealth_data, vec(header))

# Select model stats
mod_stats = Float64[]
for i in distA_stats
    if i[1] in [0.5, -0.1, -0.01]
        push!(mod_stats, i[2])
    end
end

# Compare it with data 
plot_model_vs_data(Float64.(wealth_df.value), mod_stats, ["Bottom 50%", "Top 10%", "Top 1%"], 
                    barcolor = :limegreen, title_str = "Net Wealth Distribution - Model vs Data")


#---------------------------------# 3. INCOME #--------------------------------#

# Compute Gini coefficient for income (plot Lorenz Curve)
gini_inc_pretax = compute_gini(distL * w, stat_dist, plot_curve = false)
# gini_inc_aftertax = compute_gini(distL * w - distWtax, stat_dist, plot_curve = true)

# Model: gross labor income distribution 
policy_l_incpt = policy_l .* rho_grid .* w    # diag(ρ_grid)*policy_l*w
labor_tax_policy = policy_l_incpt .- taxes.lambda_y .* policy_l_incpt .^ (1 - taxes.tau_y)

labor_tax_rate_policy = labor_tax_policy ./ policy_l_incpt 

# Compute effective average rate per income decile
# decile_shares, labor_taxes_collected, 
# Wtax_avg_rates, decile_cutoffs = analyze_income_dist(policy_l_incpt, stat_dist;
#                                                             n_deciles = 10)

# Compute gross income distribution and average income stats
policy_inc_labor_stats = compute_income_distribution_stats(stat_dist, policy_l_incpt) 
plot_dist_stats_bar(policy_inc_labor_stats, dist_type = "ptinc")  

avg_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt; 
    cutoffs = [0.5, 0.9, -0.1])

t10tob50_ratio = avg_income_stats[3][2] / avg_income_stats[1][2]
t10tob90_ratio = avg_income_stats[3][2] / avg_income_stats[2][2]

# Compute average effective rates by population decile
avg_rates_Wtax = compute_average_rate_stats(stat_dist, labor_tax_policy)
aetr_Wtax = sum(labor_tax_rate_policy .* stat_dist)
b50t10aetr_Wtax = round.([avg_rates_Wtax[1][2], avg_rates_Wtax[3][2]], digits=3)

#------------------------------# 4. CONSUMPTION #-----------------------------#

# Computing consumption tax per each state
consumption_tax_policy = policy_c .- taxes.lambda_c .* policy_c .^ (1 - taxes.tau_c) 

# A glance at the rates
cons_tax_eff_rates = consumption_tax_policy ./ policy_c

cons_tax_gini = compute_gini(consumption_tax_policy, stat_dist; plot_curve=false)
pre_tax_cons_gini = compute_gini(policy_c .+ consumption_tax_policy, stat_dist; plot_curve=false)

# Compute Kakwani Index - Gini for tax collected - Gini for pre-tax tax base
kakwani_cons_tax = cons_tax_gini - pre_tax_cons_gini

#---------------------------------# 5. LABOR #--------------------------------#

# Compute average hours worked 
avgH = sum(policy_l .* stat_dist)
println("Income process: $(rhopar.rho_prod_ar1), $(rhopar.sigma_prod_ar1)")
println("Gini for income distribution: $gini_inc_pretax")
println("Average hours worked: $avgH")
println("Capital-to-output ratio: $KtoY")    
GtoY = aggG/aggY
println("GovExp-to-output ratio: $GtoY")
# Taxation calibration results 
shareWtax = aggT_y / aggG
shareCtax = aggT_c / aggG
shareKtax = aggT_k / aggG
cons_tax_rates_min_max = round.(extrema(cons_tax_eff_rates), digits=3)
labor_tax_rates_min_max = round.(extrema(labor_tax_rate_policy), digits=3)

println("Share of revenue from labor income tax: $shareWtax")
println("Share of revenue from consumption tax: $shareCtax")
println("Share of revenue from capital return tax: $shareKtax")
println("Kakwani index for consumption tax: $kakwani_cons_tax")
println("Rates for consumption tax ranging between: $cons_tax_rates_min_max")
println("AETRs for labor income tax ranging between: $b50t10aetr_Wtax")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------------# 4. EXPORTING RESULTS #---------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#------------------------# Export calibration details #-----------------------# 

calibration_pars = String[]
values = Float64[]

# Collect parameters with their values
for (name, obj) in [("hhpar", hhpar), 
                    ("rhopar", rhopar),
                    ("fpar", fpar), 
                    ("taxes", taxes),
                    ("gpar", gpar)]
    for field in fieldnames(typeof(obj))
        push!(calibration_pars, "$name.$field")
        push!(values, getfield(obj, field))
    end
end

# Export to CSV
data = [calibration_pars values]
writedlm("./output/baseline/parameters.csv", data, ',')


#------------------------# Export steady state results #-----------------------# 

# Save to file equilibrium details 
items = Dict(
    # Calibration details
    :hhpar => hhpar, :rhopar => rhopar, :fpar => fpar, :taxes => taxes, :gpar => gpar, 
    # Equilibrium 
    :r => r, :w => w, :stat_dist => stat_dist,
    # Policy rules
    :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c,
    # Main distributions
    :distC => distC, :distK => distK, :distH => distH, :distL => distL,
    :distCtax => distCtax, :distWtax => distWtax, :distKtax => distKtax,
    # Main aggregates
    :aggC => aggC, :aggK => aggK, :aggH => aggH, :aggL => aggL, :aggG => aggG, :aggY => aggY,
    :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
    # Accuracy stats
    :excess_prod => excess_prod, :bc_max_discrepancy => bc_max_discrepancy[1],
)

for (name, mat) in items
    filepath = "./output/baseline/model_results/" * string(name) * ".txt"
    SaveMatrix(mat, filepath; overwrite=true)
end

#---------------------------# Export session details #-------------------------# 

# Print session details 
time_end = now()
session_time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(time_end) - Dates.DateTime(time_start)))
timestamp_end = Dates.format(now(), "yyyymmdd-HH_MM_SS")

print_simulation_details("./output/baseline/session_end_$(timestamp_end).txt")

