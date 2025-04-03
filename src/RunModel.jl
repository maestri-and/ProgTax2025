###############################################################################
############################## MODEL_SOLUTION.JL ##############################

############################# This script solves ##############################
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


include("Parameters.jl")
include("FirmsGov.jl")
include("AuxiliaryFunctions.jl")
include("Numerics.jl")
include("Households.jl")
include("Interpolants.jl")
include("SolvingFunctions.jl")
include("PlottingFunctions.jl")
include("../tests/TestingFunctions.jl")

# Format date for temporary outputs
ddmm = Dates.format(today(), "mm-dd")
timestamp = Dates.format(now(), "yyyymmdd-HH_MM_SS")

# Open log file
# logfile = open("output/logs/model_log_$(timestamp).txt", "w")
# redirect_stdout(logfile)
# redirect_stderr(logfile)


@info("Starting model solution...")
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

@info("Making grids...")

# Assets
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a)

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l)

# Labor productivity - Defined in model_parameters.jl
# rho_grid = rho_grid
# Extract stable distribution from transition matrix
rho_dist = find_stable_dist(pi_rho)

# Taxation parameters - baseline calibration
taxes = Taxes(0.7, 0.2, # lambda_y, tau_y, 
            0.7, 0.136, #lambda_c, tau_c,
            0.2 # tau_k
            )

# # Taxation parameters - no taxes            
# taxes = Taxes(1.0, 0.0, # lambda_y, tau_y, 
# 1.0, 0.0, #lambda_c, tau_c,
# 0.0 # tau_k
# )

# Taxes' progressivity parameters
cons_prog = range(0.0, 0.5, 2)
labor_prog = range(0.0, 0.5, 2)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------------# 2. SOLVING MODEL #-----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Solve for several combinations of taxes' progressivity
# Keeping lambdas constant 

for prl_i in eachindex(labor_prog)
    for prc_i in eachindex(cons_prog)
        # Set progressivity rates
        taxes.tau_y = labor_prog[prl_i]
        taxes.tau_c = cons_prog[prc_i]

        @info("Solving for τ_y: $(taxes.tau_y), τ_c: $(taxes.tau_c), τ_k: $(taxes.tau_k)")

        # Compute equilibrium 
        @elapsed r, w, stat_dist, valuef, policy_a, policy_l, policy_c, 
                rates, errors = ComputeEquilibrium(a_grid, rho_grid, l_grid, 
                                                    gpar, hhpar, fpar, taxes,
                                                    pi_rho, comp_params)

        # plot_heatmap_stationary_distribution(stat_dist; taxes=taxes)

        # Compute other useful distributions and aggregates
        consumption_dist, consumption_tax_dist, labor_tax_dist, 
        capital_tax_dist, aggC, aggT_c, aggT_y, aggT_k = compute_aggregates(stat_dist, policy_a, policy_c, policy_l, rho_grid, a_grid, w, r, taxes);

        # Plot rates vs errors
        # scatter(rates, errors)

        # Interpolate and return value function and policy functions
        # valuef_int, policy_a_int, policy_c_int, policy_l_int = interpolate_policy_funs(valuef, policy_a, policy_c, policy_l, rho_grid, a_grid);

        # Plot policy functions if necessary
        # plot_household_policies(valuef, policy_a, policy_l, policy_c,
        #                                  a_grid, rho_grid, taxes;
        #                                  plot_types = ["value", "assets", "labor", "consumption"],
        #                                  save_plots = false)

        # Save to file equilibrium details 
        items = Dict(:r => r, :w => w, :stat_dist => stat_dist,
            :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c,
            :consumption_dist => consumption_dist, :consumption_tax_dist => consumption_tax_dist,
            :labor_tax_dist => labor_tax_dist, :capital_tax_dist => capital_tax_dist,
            :aggC => aggC, :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
        )
        
        for (name, mat) in items
            filepath = "./output/preliminary/model_results/" * string(name) * ".txt"
            SaveMatrix(mat, filepath; overwrite=false)
        end
    end
end

# Bug with 0.5,0.5,0.5 - not converging 

# Adjust grids

#######################################################################################

# Store the VFI guess 
# SaveMatrix(V_new, "output/preliminary/V_guess_matrix_a" * "$gpar.N_a" * "_l" * "$gpar.N_l" * ".txt")
# V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")

# Save the figure
# savefig(pfa, "output/preliminary/asset_policy_len$gpar.N_a.png")

# Save the figure
# savefig(pfl, "output/preliminary/labor_policy_l$gpar.N_l" * "_a$gpar.N_a" * ".png")

# TBM - capital taxation should act only if household is saving (not on borrowing)?