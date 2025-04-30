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
gpar = GridParams(a_min, 200.000, 100, # Assets
                    0.0, 1, 50,    # Labor
                    length(rho_grid) # Productivity 
                    )

# Assets
a_gtype = "polynomial"
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = a_gtype, pol_power = 3)

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l)

# Labor productivity - Defined in Parameters.jl
# Extract stable distribution from transition matrix
rho_dist = find_stable_dist(pi_rho)

# Taxation parameters - baseline calibration
# taxes = Taxes(0.7, 0.2, # lambda_y, tau_y, 
#             0.7, 0.136, #lambda_c, tau_c,
#             0.3 # tau_k
#             )

# Taxation parameters - no taxes            
taxes = Taxes(1.0, 0.0, # lambda_y, tau_y, 
1.0, 0.0, #lambda_c, tau_c,
0.0 # tau_k
)

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

# Compute equilibrium - Newton jumping a lot for small errors!
r, w, stat_dist, valuef, policy_a, policy_l, policy_c, 
rates, errors = ComputeEquilibrium_Newton(a_grid, rho_grid, l_grid, 
                                    gpar, hhpar, fpar, taxes,
                                    pi_rho, comp_params; 
                                    prevent_Newton_jump = false)

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
# plot_household_policies(valuef, policy_a, policy_l, policy_c,
#                                  a_grid, rho_grid, taxes;
#                                  plot_types = ["value", "assets", "labor", "consumption"],
#                                  save_plots = false)

# 3D plot: labor policy function
# plot_policy_function_3d(policy_l, a_grid, rho_grid; policy_type="labor")

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

plot_wealth_dist_bar(distA_stats)

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
compute_gini(stat_dist, distL * w, plot_curve = true)


#---------------------------------# 3. LABOR #--------------------------------#

# Compute average hours worked 
avgH = sum(policy_l .* stat_dist)
