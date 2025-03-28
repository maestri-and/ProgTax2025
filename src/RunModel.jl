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
# Extract stable distribution from transition matrix
rho_dist = find_stable_dist(pi_rho)

# # Taxation parameters - baseline calibration
taxes = Taxes(0.7, 0.2, # lambda_y, tau_y, 
            0.7, 0.136, #lambda_c, tau_c,
            0.0 # tau_k
            )

taxes = Taxes(1.0, 0.0, # lambda_y, tau_y, 
1.0, 0.0, #lambda_c, tau_c,
0.0 # tau_k
)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------------# 2. SOLVING MODEL #-----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function ComputeEquilibrium(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params
)
    #--- Initial bounds for interest rate (r) for bisection method
    r_low = -fpar.delta
    r_high = 1 / hhpar.beta - 1
    r_mid = (r_low + r_high) / 2
    ω = 0.5  # Weight for dampened update of r

    #--- Wage implied by firm's FOC given r
    opt_wage(r) = (1 - fpar.alpha) * fpar.tfp *
                  ((fpar.alpha * fpar.tfp / (r + fpar.delta)) ^ (fpar.alpha / (1 - fpar.alpha)))
    w = opt_wage(r_mid)

    #--- Begin equilibrium loop
    @elapsed for iter in 1:comp_params.ms_max_iter

        ###### 1. Household Problem ######
        (hh_labor_taxes, hh_consumption, hh_consumption_tax,
         opt_c_FOC, opt_l_FOC, valuef, policy_a,
         policy_l, policy_c) = SolveHouseholdProblem(
             a_grid, rho_grid, l_grid, gpar, w, r_mid, taxes,
             hhpar, pi_rho, comp_params
         )

        ###### 2. Stationary Distribution ######
        stat_dist = stationary_distribution(
            a_grid, pi_rho, policy_a, gpar;
            tol = 1e-10, max_iter = 10_000
        )

        ###### 3. Aggregates ######
        asset_supply = sum(stat_dist * a_grid)  # asset by productivity
        labor_supply = sum(stat_dist .* policy_l)
        consumption_demand = sum(stat_dist .* policy_c)

        ###### 4. Firm's Capital Demand from FOC ######
        asset_demand = ((fpar.alpha * fpar.tfp) / (r_mid + fpar.delta)) ^
                       (1 / (1 - fpar.alpha)) * labor_supply

        ###### 5. Check for Market Clearing ######
        K_error = asset_demand - asset_supply

        println("Iter $iter: r = $(round(r_mid, digits=5)), w = $(round(w, digits=5)), K_supply = $(round(asset_supply, digits=5)), K_demand = $(round(asset_demand, digits=5)), error = $(round(K_error, digits=5))")

        if abs(K_error) < comp_params.ms_tol
            println("✅ Equilibrium found: r = $r_mid, w = $w after $iter iterations")
            return r_mid, w, stat_dist, valuef, policy_a, policy_l, policy_c
        end

        ###### 6. Bisection Update of Interest Rate ######
        if K_error > 0
            r_low = r_mid  # Excess demand → raise r
        else
            r_high = r_mid  # Excess supply → lower r
        end

        r_new = ω * r_mid + (1 - ω) * ((r_low + r_high) / 2)
        r_mid = r_new
        w = opt_wage(r_mid)
    end

    error("❌ Equilibrium not found within max iterations.")
end

@elapsed r, w, stat_dist, valuef, policy_a, policy_l, policy_c = ComputeEquilibrium(a_grid, rho_grid, l_grid, 
                                                                            gpar, hhpar, fpar, taxes,
                                                                            pi_rho, comp_params
                                                                        )

heatmap(stat_dist, xlabel="a index", ylabel="ρ index")

# Interpolate and return value function and policy functions
valuef_int, policy_a_int, policy_c_int, policy_l_int = interpolate_policy_funs(valuef, policy_a, policy_c, policy_l, rho_grid, a_grid);

# Plot policy functions if necessary
plot_household_policies(valuef, policy_a, policy_l, policy_c,
                                 a_grid, rho_grid, taxes;
                                 plot_types = ["value", "assets", "labor", "consumption"],
                                 save_plots = false)


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