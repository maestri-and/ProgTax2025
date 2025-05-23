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
baseline_path = "output/equilibria/baseline/model_results"
baseline = get_model_results(baseline_path)

# Extract baseline taxes and government expenditure
b_taxes = Taxes(baseline.taxes[1]...)
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
#-----------------# 3. ITERATE OVER DIFFERENT TAX REGIMES  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#--------------# Change in tau_y: solve for G-equivalent tau_c #--------------#

# Fix number of different regimes to simulate
n_sim = nthreads() * 3

# Create matrix of Taxes structs for each simulation
regimes = [deepcopy(b_taxes) for t in 1:n_sim]

# Generate new labor income tax progressivity parameters
# Range: -20% + 20%
perc_vars = collect(range(-0.2, 0.2, n_sim + 1))
var_coeffs = 1 .+ perc_vars[perc_vars .!= 0] # Remove baseline

for t in 1:n_sim
    regimes[t].tau_y = round(b_taxes.tau_y * var_coeffs[t], digits = 5)
end

# Clean and prepare folders
GC.gc()
foreach(x -> rm(x; force=true, recursive=true), readdir("output/equilibria/equivalent_regimes", join=true))
eqr_paths = [joinpath("output/equilibria/equivalent_regimes", "eqr$i") for i in 1:n_sim]
eqr_res_paths = [joinpath(eqr_paths[i], "model_results") for i in 1:n_sim]

# Prepare containers
eq_regimes = deepcopy(regimes)
eq_rates = zeros(length(regimes))

# Launch tasks
@info("Launching parallelised tax regime search...")
tasks = []

for i in 1:n_sim
    push!(tasks, Threads.@spawn begin
        @info "[Thread $(threadid())] Starting simulation #$i"

        #-----# Run model #-----#
        t_taxes = deepcopy(regimes[i])
        new_tau_c, r_eq, w_eq, stat_dist, valuef, policy_a, policy_l, policy_c = TwoLevelEquilibriumNewton(
            a_grid, rho_grid, l_grid,
            gpar, hhpar, fpar, t_taxes,
            pi_rho, comp_params, b_aggG
        )

        # Fix new taxes parameter
        t_taxes.tau_c = new_tau_c

        #-----# Aggregate results #-----#
        distC, distK, distH, distL,
        distCtax, distWtax, distKtax, 
        aggC, aggK, aggH, aggL, aggG, aggY,
        aggT_c, aggT_y, aggT_k, 
        excess_prod, bc_max_discrepancy = compute_aggregates_and_check(stat_dist, policy_a, policy_c, 
                                                                            policy_l, rho_grid, a_grid, w_eq, r_eq, t_taxes;
                                                                            raise_bc_error = false, 
                                                                            raise_clearing_error = false);        

        #-----# Save results #-----#
        eq_regimes[i].tau_c = new_tau_c
        eq_rates[i] = r_eq

        # Create output folders
        mkpath(eqr_paths[i])
        mkpath(eqr_res_paths[i])

        # Save to file
        # Tax regime search
        filepath = joinpath(eqr_paths[i], "search_results.txt")
        WriteTaxSearchResults(t_taxes, r_eq, aggG, filepath)
        # Model results
        items = Dict(
            # Equilibrium 
            :r => r_eq, :w => w_eq, :stat_dist => stat_dist,
            # Policy rules
            :policy_a => policy_a, :policy_l => policy_l, :policy_c => policy_c, :valuef => valuef,
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
            filepath = joinpath(eqr_res_paths[i], string(name) * ".txt")
            SaveMatrix(mat, filepath; overwrite=false, taxes = t_taxes)
        end

    end)
end

# Wait for all threads to finish
wait.(tasks)


#---------------------------# Export session details #-------------------------# 

# Print session details 
time_end = now()
session_time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(time_end) - Dates.DateTime(time_start)))
timestamp_end = Dates.format(now(), "yyyymmdd-HH_MM_SS")

print_eq_regime_search_session_details(eq_regimes, eq_rates, b_taxes,
                                        "./output/equilibria/equivalent_regimes/session_end_$(timestamp_end).txt")