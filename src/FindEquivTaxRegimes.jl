###############################################################################
########################### FINDEQUIVTAXREGIMES.JL ############################

########### This script iterates the benchmark ProgTax(2025) model ############
################# to find tax-revenue-equivalent tax regimes ##################

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
baseline_path = "output/baseline/model_results"
baseline = get_model_results(baseline_path)

# Extract baseline taxes and government expenditure
b_taxes = Taxes(baseline.taxes[1]...)
b_aggG = baseline.aggG[1]


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 2. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

@info("Making grids...")

# Define grid parameters
gpar = GridParams(-1.000, 200.000, 100, # Assets
                    0.0, 1, 50,    # Labor
                    length(rho_grid) # Productivity 
                    )

# Assets
a_gtype = "polynomial"
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = a_gtype, pol_power = 3.5)
# a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = "log")

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------# 3. ITERATE OVER DIFFERENT TAX REGIMES  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#--------------# Change in tau_y: solve for G-equivalent tau_c #--------------#

# Fix number of different regimes to simulate
n_sim = 2

# Create matrix of Taxes structs for each simulation
regimes = [deepcopy(b_taxes) for t in 1:n_sim]

# Generate new labor income tax progressivity parameters
# Range: -20% + 20%
var_coeffs = 1 .+ collect(range(-0.2, 0.2, n_sim))

for t in 1:n_sim
    regimes[t].tau_y = round(b_taxes.tau_y * var_coeffs[t], digits = 5)
end

# # Clean tax_regime_search folder and create new subfolders
# foreach(x -> rm(x; force=true, recursive=true), readdir("output/tax_regime_search", join=true))
# eqr_paths = [joinpath("output/tax_regime_search", "eqr$i") for i in 1:n_sim]
# eqr_res_paths = [joinpath("$(eqr_paths[i])", "model_results") for i in 1:n_sim]

# # Create arrays to allocate results 
# eq_regimes = deepcopy(regimes)
# eq_rates = zeros(length(regimes))
# eq_G = zeros(length(regimes))

# # Find equivalent tax regime 
# @info("Launching parallelised tax regime search...")

# @elapsed @threads for i in 1:n_sim
#     # Create new subdirectories 
#     mkpath(eqr_paths[i])
#     mkpath(eqr_res_paths[i])

#     # Get relevant tax regime
#     t_taxes = deepcopy(regimes[i])

#     # Solve model 
#     new_tau_c, new_r, eq_exp = TwoLevelEquilibriumNewton(
#         a_grid, rho_grid, l_grid,
#         gpar, hhpar, fpar, t_taxes,
#         pi_rho, comp_params, b_aggG
#         )  
        
#     # Assign to G-equivalent regime struct and vector
#     t_taxes.tau_c = new_tau_c
#     eq_regimes[i].tau_c = new_tau_c
#     eq_rates[i] = new_r
#     eq_G[i] = eq_exp

#     # Print search results in .txt format
#     filepath = joinpath("$(eqr_paths[i])", "search_results.txt")
#     WriteTaxSearchResults(t_taxes, new_r, filepath::String)

# end


# Clean and prepare folders
GC.gc()
foreach(x -> rm(x; force=true, recursive=true), readdir("output/tax_regime_search", join=true))
eqr_paths = [joinpath("output/tax_regime_search", "eqr$i") for i in 1:n_sim]
eqr_res_paths = [joinpath(eqr_paths[i], "model_results") for i in 1:n_sim]

# Prepare containers
eq_regimes = deepcopy(regimes)
eq_rates = zeros(length(regimes))
eq_G = zeros(length(regimes))

# Launch tasks
@info("Launching parallelised tax regime search...")
tasks = []

for i in 1:n_sim
    push!(tasks, Threads.@spawn begin
        @info "[Thread $(threadid())] Starting simulation #$i"

        # Create output folders
        mkpath(eqr_paths[i])
        mkpath(eqr_res_paths[i])

        # Run model
        t_taxes = deepcopy(regimes[i])
        new_tau_c, new_r, eq_exp = TwoLevelEquilibriumNewton(
            a_grid, rho_grid, l_grid,
            gpar, hhpar, fpar, t_taxes,
            pi_rho, comp_params, b_aggG
        )

        # Save results
        t_taxes.tau_c = new_tau_c
        eq_regimes[i].tau_c = new_tau_c
        eq_rates[i] = new_r
        eq_G[i] = eq_exp

        # Save to file
        filepath = joinpath(eqr_paths[i], "search_results.txt")
        WriteTaxSearchResults(t_taxes, new_r, eq_exp, filepath)
    end)
end

# Wait for all threads to finish
wait.(tasks)


#---------------------------# Export session details #-------------------------# 

# Print session details 
time_end = now()
session_time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(time_end) - Dates.DateTime(time_start)))
timestamp_end = Dates.format(now(), "yyyymmdd-HH_MM_SS")

print_tax_regime_search_session_details(regimes, b_taxes,
                                        "./output/tax_regime_search/session_end_$(timestamp_end).txt")