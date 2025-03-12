###############################################################################
############################### INTERPOLANTS.JL ###############################

############## This script defines interpolations used to solve ###############
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

using LinearAlgebra
using Base.Threads
using Interpolations
using StatsBase
using Dierckx
using Optim

include("Parameters.jl")
include("Numerics.jl")
include("Households.jl")
include("AuxiliaryFunctions.jl")


###############################################################################
     ######## 1. INTERPOLATE CONSUMPTION FOR GIVEN (l, ρ, a, a') #########
###############################################################################

# ========================================================
# Build the interpolant: a cubic spline mapping x -> c
# ========================================================

function interp_consumption(hh_consumption, hh_consumption_plus_tax)
    ##-----------------------------------------------------------------------##
    # Interpolating the Feldstein relationship between consumption
    # and total consumption expenditure, using 
    # c + T_c(c) = k = y - T_y(y) + (1 + r)a - a'
    # for specific combination of Tau_y and Tau_c
    ##-----------------------------------------------------------------------##

    # Compute exact solution for coarse grid through root finding
    # _, hh_consumption, _, hh_consumption_plus_tax = compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, Tau_y, Tau_c, taxes)

    # Flatten 4D hh_consumption and hh_consumption_plus_tax into vectors:
    x_data = vec(hh_consumption_plus_tax)
    y_data = vec(hh_consumption)

    # Sort the data by x_data:
    perm = sortperm(x_data)
    x_sorted = x_data[perm]
    y_sorted = y_data[perm]

    # Create the interpolant for irregular data

    # Deduplicate knots
    Interpolations.deduplicate_knots!(x_sorted, move_knots=true)
    
    # 1. Linear Interpolation
    # 2. Fritsch-Carlson Monotonic Interpolation
    itp = interpolate((x_sorted,), y_sorted, Gridded(Linear()))
    # itp2 = interpolate(x_sorted, y_sorted, FritschCarlsonMonotonicInterpolation()) # 4x slower

    return itp
end


###############################################################################
  ######## 2. INTERPOLATE CONTINUATION VALUE FOR GIVEN STATE (ρ, a) #########
###############################################################################

function interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)
    # This function interpolates the continuation value in the Value Function
    # for each given state in the next period, given today's state
    # cont[ρ, j] = Σ_{ρ'} π(ρ,ρ') V_guess(ρ', j)
    cont = pi_rho * V_guess   # (N_rho, N_a)
    itp_cont = extrapolate(interpolate((rho_grid, a_grid), cont, Gridded(Linear())), Interpolations.Flat())
    # itp_cont = Spline2D(rho_grid, a_grid, cont) #Dierckx spline - ~50 times slower than linear
    cont_interp = (a_prime, rho) -> itp_cont(a_prime, rho) 
    
    return itp_cont, cont_interp
end

###############################################################################
      ######## 3. INTERPOLATE OPTIMAL LABOR FOR GIVEN (ρ, a, a') #########
###############################################################################

function interp_opt_labor(V_guess, pi_rho, rho_grid, a_grid)
    return nothing
end