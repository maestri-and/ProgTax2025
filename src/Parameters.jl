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


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------# 1. DEFINE STRUCTURAL PARAMETERS #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Households parameters
struct HHParams
    beta         ::Float64        # Discount factor
    rra          ::Float64        # Relative risk-aversion coefficient
    phi          ::Float64        # Psi parameter - relative disutility of labor - Ferriere et al. 2023
    frisch       ::Float64        # Frisch elasticity of substitution - Ferriere et al. 2023
end

hhpar = HHParams(
    0.96,         # Discount factor    
    2.000,        # Relative risk-aversion coefficient
    85.00,        # Psi parameter - relative disutility of labor - Ferriere et al. 2023
    2.5           # Frisch elasticity of substitution - Ferriere et al. 2023
)

# Labor productivity - Li(2013) - TBM
rho_grid = [0.1805, 0.3625, 0.8127, 1.8098, 3.8989, 8.4002, 18.0980]

# Testing other grids
rho_grid1 = [1.1805, 1.3625, 1.8127, 2.8098, 3.8989, 8.4002, 18.0980]
rho_grid2 = [1.1805, 2.3625, 5.8127, 6.8098, 8.8989, 12.4002, 18.0980]
rho_grid3 = [1.1805, 2.3625, 5.8127, 7.8098, 10.8989, 15.4002, 18.0980]
rho_grid4 = [2.1805, 4.3625, 7.8127, 10.8098, 15.8989, 21.4002, 28.0980]
rho_grid5 = exp.(range(log(3), log(20), length=7))    # Spread on log scale
rho_grid6 = range(2, 20, length=7)                    # Uniformly spaced


# Labor productivity transition matrix
pi_rho = [
    0.9687 0.0313 0 0 0 0 0;
    0.0445 0.8620 0.0935 0 0 0 0;
    0 0.0667 0.9180 0.0153 0 0 0;
    0 0 0.0666 0.8669 0.0665 0 0;
    0 0 0 0.1054 0.8280 0.0666 0;
    0 0 0 0 0.1235 0.8320 0.0445;
    0 0 0 0 0 0.2113 0.7887
]


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

# Firm parameters
struct FirmParams
    alpha        ::Float64        # Capital share of income (Cobb-Douglas)
    delta        ::Float64        # Depreciation rate
    tfp          ::Float64        # Total factor productivity
end

fpar = FirmParams(1/3, 0.06, 1) # As in Ferriere et al. 2023

# Government parameters


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


gpar = GridParams(-1.000, 80.000, 50, # Assets
                    0.0, 1, 50,    # Labor
                    length(rho_grid) # Productivity 
                    )

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