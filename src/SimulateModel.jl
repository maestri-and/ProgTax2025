###############################################################################
############################## SIMULATEMODEL.JL ###############################

########### This script simulates the benchmark ProgTax(2025) model ###########
############# to find multiple equilibria and stores the results ##############

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

# Format date for temporary outputs
ddmm = Dates.format(today(), "mm-dd")
time_start = now()
timestamp_start = Dates.format(now(), "yyyymmdd-HH_MM_SS")

# Open log file
# logfile = open("output/logs/model_log_$(timestamp_start).txt", "w")
# redirect_stdout(logfile)
# redirect_stderr(logfile)


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
# a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = "log")

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l)

# Labor productivity - Defined in Parameters.jl
# rho_grid = rho_grid
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

# Taxes' progressivity parameters
cons_prog = range(0.0, 0.5, 2)
labor_prog = range(0.0, 0.5, 2)

# cons_prog = range(0.0, 0.5, 21)
# labor_prog = range(0.0, 0.5, 21)


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

        @info("Solving for λ_y: $(taxes.lambda_y),  τ_y: $(taxes.tau_y), λ_c: $(taxes.lambda_c), τ_c: $(taxes.tau_c), τ_k: $(taxes.tau_k)")

        # Compute equilibrium - Newton jumping a lot for small errors!
        r, w, stat_dist, valuef, policy_a, policy_l, policy_c, 
        rates, errors = ComputeEquilibrium_Newton(a_grid, rho_grid, l_grid, 
                                            gpar, hhpar, fpar, taxes,
                                            pi_rho, comp_params; 
                                            prevent_Newton_jump = false)

        # Plot stationary distribution 
        # plot_heatmap_stationary_distribution(stat_dist; taxes=taxes)
        # plot_density_by_productivity(stat_dist, a_grid, gpar; rho_grid=nothing)

        # Compute other useful distributions and aggregates
        distC, distK, distH, distL,
        distCtax, distWtax, distKtax, 
        aggC, aggK, aggH, aggL, aggG, aggY,
        aggT_c, aggT_y, aggT_k, 
        excess_prod, bc_max_discrepancy = compute_aggregates_and_check(stat_dist, policy_a, policy_c, 
                                                                            policy_l, rho_grid, a_grid, w, r, taxes;
                                                                            raise_bc_error = false, raise_clearing_error = false);        

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

        # Save to file equilibrium details 
        items = Dict(:r => r, :w => w, :stat_dist => stat_dist,
            :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c,
            :distC => distC, :distK => distK, :distH => distH, :distL => distL,
            :distCtax => distCtax, :distWtax => distWtax, :distKtax => distKtax,
            :aggC => aggC, :aggK => aggK, :aggH => aggH, :aggL => aggL, :aggG => aggG, :aggY => aggY,
            :aggT_c => aggT_c, :aggT_y => aggT_y, :aggT_k => aggT_k,
            :excess_prod => excess_prod, :bc_max_discrepancy => bc_max_discrepancy[1]
        )
        
        for (name, mat) in items
            filepath = "./output/preliminary/model_results/" * string(name) * ".txt"
            SaveMatrix(mat, filepath; overwrite=false)
        end
    end
end

# Print session details 
time_end = now()
session_time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(time_end) - Dates.DateTime(time_start)))
timestamp_end = Dates.format(now(), "yyyymmdd-HH_MM_SS")

print_simulation_details("./output/preliminary/model_results/session_end_$(timestamp_end).txt")

@info("Solved all steady states!")

#######################################################################################
