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

# Households

# Labor productivity - Li(2013) - TBM
rho_grid = [0.1805, 0.3625, 0.8127, 1.8098, 3.8989, 8.4002, 18.0980]
N_rho = length(rho_grid)

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


# Government's decision problem parameters

sigma        = 2.000        # Risk aversion
psi          = 0.050        # Probability of re-entry in capital markets
tau          = 0.410        # Tax revenues over GDP - 
g_star       = 0.680        # Nondiscretionary spending over tax revenues


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------# 2. DEFINE COMPUTATIONAL PARAMETERS #-------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Assets
a_min        = 1.000        # Lower bound for asset space
a_max        = 5.000        # Upper bound for asset space
N_a          = 100          # Number of points for asset grid

# Labor
l_min        = 0.000        # Lower bound for labor space
l_max        = 1.000        # Upper bound for labor space
N_l          = 10          # Number of points for labor grid

# Taxation
# First minimal definition - TBM and vectorised

struct Taxes
    lambda_y::Float64
    tau_y::Vector{Float64}
    N_tau_y::Int
    lambda_c::Float64
    tau_c::Vector{Float64}
    N_tau_c::Int

    function Taxes(lambda_y::Float64, tau_y::Vector{Float64}, lambda_c::Float64, tau_c::Vector{Float64})
        new(lambda_y, tau_y, length(tau_y), lambda_c, tau_c, length(tau_c))
    end
end


# tau_c_min   = 0            # Minimum degree of progressivity for consumption tax - Negative = Regressive, 0 = linear
# tau_c_max   = 1            # Maximum degree of progressivity for consumption tax
# N_tau_c     = 100          # Number of grid points for consumption tax progressivity

# tau_y_min   = 0            # Minimum degree of progressivity for labor income tax - Negative = Regressive, 0 = linear
# tau_y_max   = 1            # Maximum degree of progressivity for labor income tax
# N_tau_y     = 100          # Number of grid points for labor income tax progressivity

# First minimal definition
lambda_y = 0.7
tau_y = [0.136, 0.2]
lambda_c = 0.7
tau_c = [0.136, 0.2]

taxes = Taxes(lambda_y, tau_y, lambda_c, tau_c)


### - ### OLD ### - ###

state_ex    = 3;              # Number of exogenous state variables
N           = 5;              # Number of points for each exogenous state in value function
N_ex        = N^state_ex;          # Numper of grid points for value function approximation
max_iter    = 550;          # Maximum number of iterations

convergence_q = 0.999;      # Smoothing of pricing schedule
convergence_v = 0.00;       # Smoothing of value function
val_lb        = 0.02;       # Lower bound for value function
lwb           = 0.75;       # Lower bound for debt grid
uwb           = 1.25;       # Upper bound for debt grid
start_conv    = 1;    
tolerance     = 10^(-4);

###############################################################################
#                        Bounds for exogenous state variables 
###############################################################################
