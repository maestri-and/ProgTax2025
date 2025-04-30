###############################################################################
############################### CALIBRATION.JL ################################

###################### This script is used to calibrate #######################
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
using QuantEcon

include("AuxiliaryFunctions.jl")
include("PlottingFunctions.jl")
include("Parameters.jl")
include("AnalysisFunctions.jl")
include("Numerics.jl")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. HOUSEHOLDS' PRODUCTIVITY PROCESS #--------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Calibrating productivity process using Rouwenhorst - Copied in Parameters.jl
# AR(1) persistency parameter drawn from LANGOT, MALMBERG, TRIPIER, HAIRAULT, 2023
# AR(1) volatility calibrated to match Gini coefficient for income distribution
rho_prod_ar1 = 0.966
sigma_prod_ar1 = 0.4786
n_prod_ar1 = 7
mean_prod_ar1 = 0

markov_rho = rouwenhorst(n_prod_ar1, rho_prod_ar1, sigma_prod_ar1, mean_prod_ar1)

pi_rho = markov_rho.p
pi_states = exp.(collect(markov_rho.state_values))

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 2. CALIBRATING TAX REGIMES #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#-#-#-#-#-#-#-#-#-#-#-# a. Calibrating Labor Income Tax #-#-#-#-#-#-#-#-#-#-#-#

# Interpolate target labor income tax
# Taxes - construct personal income tax
# Define tax brackets: ((lower_bound, upper_bound), rate)
# French Income Tax Brackets, 2025 - Family coefficient = 1
# Source: https://www.service-public.fr/particuliers/actualites/A18045
brackets = [
    ((0.0,      11_497.0), 0.00),
    ((11_497.0, 29_315.0), 0.11),
    ((29_315.0, 83_823.0), 0.30),
    ((83_823.0, 180_294.0), 0.41),
    ((180_294.0, Inf),     0.45)
]

# Compute effective tax rate and plot
target_eff_rate = compute_effective_tax(brackets; output = :rate, plot = true,
                                        graph_title = "Statutory Effective Tax Rate - France, 2025 (Family coefficient = 1)")

# Compute total tax amount and plot
target_tax = compute_effective_tax(brackets; output = :taxes, plot = true,
                                   graph_title = "Statutory Income Tax Liability - France, 2025 (Family coefficient = 1)")

# De-mean