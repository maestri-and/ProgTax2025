###############################################################################
############################# RESULTS_ANALYSIS.JL #############################

######### This script analyses the results produced by simulations of #########
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
using DataFrames
using CairoMakie
using Plots
using BenchmarkTools
using Dates

include("AuxiliaryFunctions.jl")
include("PlottingFunctions.jl")
include("Parameters.jl")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 1. IMPORTING MODEL RESULTS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Get steady state values for multiple tax structs
folderpath = "output/preliminary/model_results/-1to60_len200_log/"
ss = get_model_results(folderpath)

# Compute variables
# Government expenditure
ss[:, "aggG"] = ss.aggT_c .+ ss.aggT_y .+ ss.aggT_k

# Aggregate capital 
ss.aggK .= 0.0
for i in 1:nrow(ss)
    ss.aggK[i] = sum(ss.policy_a[i] .* ss.stat_dist[i])
end

# Aggregate labor 
ss.aggH .= 0.0
for i in 1:nrow(ss)
    ss.aggH[i] = sum(ss.policy_l[i] .* ss.stat_dist[i])
end

# Aggregate output
# Production side 
ss.aggY = fpar.tfp .* ss.aggL .^ (1 - fpar.alpha) .* ss.aggK .^ fpar.alpha
# Income side
ss.aggLaborIncome .= 0.0
for i in 1:nrow(ss)
    ss.aggLaborIncome[i] = sum(ss.w[i] .* rho_grid .* policy_l[i])
end
ss.aggY_is = ss.aggLaborIncome .+ ss.r .* ss.aggK

# Investment 
ss.aggI = fpar.delta .* ss.aggK

# Goods market clearing 
ss.GMerror = ss.aggY .- ss.aggC .- ss.aggI - ss.aggG
ss.GMerror_frac = ss.GMerror ./ ss.aggY

aggregates = select(ss, names(ss)[1:5]..., "aggK", "aggL", "aggC", "aggI", "aggG", "aggY", "GMerror", "GMerror_frac")


# Gini 


# Plot consumption by progressivity 
plot_aggregate_surface(ss.aggC, ss.tau_c, ss.tau_y;
                        zlabel = "Consumption",
                        title_text = "Aggregate consumption by tax progressivity",
                        cmap = :avocado,
                        azimuth = 5π/4)


# Government expenditure
plot_aggregate_surface(ss.G, ss.tau_c, ss.tau_y;
                    zlabel = "Government expenditure",
                    title_text = "Government expenditure by tax progressivity",
                    cmap = :linear_bgyw_20_98_c66_n256)

# Savings/Capital
plot_aggregate_surface(ss.aggK, ss.tau_c, ss.tau_y;
                    zlabel = "Aggregate savings",
                    title_text = "Aggregate savings by tax progressivity",
                    cmap = :summer)


# Interest rate                    
plot_aggregate_surface(ss.r, ss.tau_c, ss.tau_y;
                    zlabel = "Interest rate",
                    title_text = "Equilibrium rate by tax progressivity",
                    cmap = :heat,
                    azimuth = 5π/4)

# Wage
plot_aggregate_surface(ss.w, ss.tau_c, ss.tau_y;
                    zlabel = "Wage",
                    title_text = "Equilibrium wage by tax progressivity",
                    cmap = :haline)

# Interest rate
