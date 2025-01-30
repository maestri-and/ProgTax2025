###############################################################################
############################### PARAMETERS.JL #################################

############## This script defines and assigns the parameters of ##############
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


###############################################################################
#                               Import libraries
###############################################################################

using LinearAlgebra

###############################################################################
#                        Define structural parameters
###############################################################################

# Government's decision problem parameters

sigma   = 2.000                 # Risk aversion
psi     = 0.050                 # Probability of re-entry in capital markets
tau     = 0.410                 # Tax revenues over GDP
g_star       = 0.680            # Nondiscretionary spending over tax revenues


###############################################################################
#                        Define computational parameters
###############################################################################

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
