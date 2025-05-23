###############################################################################
############################### PARAMETERS.JL #################################

############## This script defines and assigns the parameters of ##############
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


using LinearAlgebra
using QuantEcon

include("HouseholdsFirmsGov.jl")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------# 1. DEFINE STRUCTURAL PARAMETERS #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Labor productivity process
struct prodAR1Params
    rho_prod_ar1::Float64   # Persistency of AR(1) income process
    sigma_prod_ar1::Float64 # Volatility of AR(1) income process
    n_prod_ar1::Int64       # Number of states 
    mean_prod_ar1:: Float64 # Mean of AR(1) process
end

# Calibrating productivity process using Rouwenhorst 
# AR(1) process parameters from Cahn et al. 2024
rhopar = prodAR1Params(0.9183,
                       0.2976,
                       7,
                       0.0)

markov_rho = rouwenhorst(rhopar.n_prod_ar1, rhopar.rho_prod_ar1, rhopar.sigma_prod_ar1, rhopar.mean_prod_ar1)

pi_rho = markov_rho.p
rho_grid = exp.(collect(markov_rho.state_values))
uncond_var = rhopar.sigma_prod_ar1^2 / (1 - rhopar.rho_prod_ar1^2)

# Household parameters
struct HHParams
    beta         ::Float64        # Discount factor
    rra          ::Float64        # Relative risk-aversion coefficient
    dis_labor    ::Float64        # Psi parameter - relative disutility of labor 
    inv_frisch   ::Float64        # Inverse of Frisch elasticity of substitution 
end

hhpar = HHParams(
    0.9675,      # Discount factor    
    1.500,       # Relative risk-aversion coefficient
    85.00,       # Phi parameter - relative disutility of labor
    2.5          # Inverse of Frisch elasticity of substitution 
)

# Taxation
mutable struct Taxes
    # Parameters for tax system

    # Labor income tax - Feldstein
    lambda_y::Float64
    tau_y::Float64

    # Consumption tax - Feldstein
    lambda_c::Float64
    tau_c::Float64

    # Capital tax - Linear
    tau_k::Float64
end

# Outer constructor for a 5-element tuple
Taxes(t::NTuple{5, Float64}) = Taxes(t...)

# Firm parameters
struct FirmParams
    alpha        ::Float64        # Capital share of income (Cobb-Douglas)
    delta        ::Float64        # Depreciation rate
    tfp          ::Float64        # Total factor productivity
end

fpar = FirmParams(1/3, 0.084, 1) 

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------# 2. DEFINE COMPUTATIONAL PARAMETERS #-------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Grid parameters
struct GridParams
    # Assets 
    a_min        ::Float64        # Lower bound for asset space
    a_max        ::Float64        # Upper bound for asset space
    N_a          ::Int64          # Number of points for asset grid

    # Labor
    l_min        ::Float64        # Lower bound for labor space
    l_max        ::Float64        # Upper bound for labor space
    N_l          ::Int64          # Number of points for labor grid

    # Productivity
    N_rho        ::Int64          # Number of points for productivity grid
end

# Compute natural borrowing limit
# Derived such that low-productivity-and-wealth household
# can pay proceedings from her debt even with a very high interest rate
a_min_r = 0.1
a_min_l_max = 0.8
a_min = - rho_grid[1] * cd_implied_opt_wage(a_min_r) * a_min_l_max / a_min_r
# a_min = -0.50 # Fix to test Gini 

# Iterations and computations
struct CompParams
    vfi_max_iter    ::Int64
    vfi_tol         ::Float64
    ms_max_iter     ::Int64
    ms_tol          ::Float64
end


comp_params = CompParams(
    3000, 10^(-4),      # VFI parameters
    500, 10^(-6)      # Model solution parameters
)