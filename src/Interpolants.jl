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
include("HouseholdsFirmsGov.jl")
include("AuxiliaryFunctions.jl")


###############################################################################
     ######## 1. INTERPOLATE CONSUMPTION FOR GIVEN (l, ρ, a, a') #########
###############################################################################

# ========================================================
# Build the interpolant: a cubic spline mapping x -> c
# ========================================================

function interp_consumption(hh_consumption, hh_consumption_plus_tax; piecewise = true)
    ##-----------------------------------------------------------------------##
    # Interpolating the Feldstein relationship between consumption
    # and total consumption expenditure, using 
    # c + T_c(c) = k = y - T_y(y) + (1 + (1 - tau_k)r)a - a'
    # for specific combination of Tau_y and Tau_c
    ##-----------------------------------------------------------------------##

    # Compute exact solution for coarse grid through root finding
    # _, hh_consumption, _, hh_consumption_plus_tax = compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, Tau_y, Tau_c, taxes)

    # Flatten 4D hh_consumption and hh_consumption_plus_tax into vectors:
    x_data = vec(hh_consumption_plus_tax)
    y_data = vec(hh_consumption)

    if piecewise == true
        # Identify positive region (interpolation needed)
        positive_mask = x_data .> 0

        # Get data for interpolation (only for positive values)
        x_pos = x_data[positive_mask]
        y_pos = y_data[positive_mask]

        # Sort the data (needed for interpolation)
        perm = sortperm(x_pos)
        x_sorted = x_pos[perm]
        y_sorted = y_pos[perm]

            # Deduplicate knots
        Interpolations.deduplicate_knots!(x_sorted, move_knots=true)

        # Create linear interpolation for positive values
        itp = interpolate((x_sorted,), y_sorted, Gridded(Linear()))
        itp_extrap = extrapolate(itp, Interpolations.Flat())  # Avoids artificial smoothing

        # Define the final piecewise function
        function piecewise_interp(x)
            if x ≤ 0
                return x  # Identity function for x <= 0
            else
                return itp_extrap(x)  # Interpolation for x > 0
            end
        end

        return piecewise_interp
    else
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
end



###############################################################################
              ######## 2. GENERIC PIECEWISE INTERPOLATION #########
###############################################################################

function piecewise_1D_interpolation(x, y; spline=false, return_threshold=false)
    """
    Performs piecewise 1D interpolation, handling cases where y jumps suddenly to -Inf.

    Args:
        x               : Vector of x values (independent variable)
        y               : Vector of y values (dependent variable, may contain -Inf)
        spline          : Boolean, if true uses cubic splines (Dierckx), otherwise uses linear interpolation
        return_threshold: Boolean, if true returns both the interpolant and the detected threshold.

    Returns:
        - If `return_threshold = false`: Returns a piecewise interpolation function.
        - If `return_threshold = true`: Returns (interpolant function, threshold value).
    """

    # Step 1: Identify valid (finite) indices where y is not -Inf
    valid_indices = findall(isfinite, y)

    if isempty(valid_indices)
        # Case: All y values are -Inf → No valid data for interpolation
        threshold_x = minimum(x)  # Set threshold to the lower bound of x
        # @warn "No valid y values found for interpolation! Returning constant -Inf function."

        # Define a function that always returns -Inf
        piecewise_interp0 = x_query -> -Inf

        if return_threshold
            return piecewise_interp0, threshold_x
        else
            return piecewise_interp0
        end
    elseif length(valid_indices) == 1
        # Case 2: Only one valid (finite) y value → No interpolation possible
        threshold_x = minimum(x)  # Set threshold to the lower bound of x
        x_single = x[valid_indices[1]]
        y_single = y[valid_indices[1]]

        # @warn "Only one valid y value found! Returning a step function."

        # Define a function that returns y_single for x_single, and -Inf otherwise
        function piecewise_interp1(x_query)
            return x_query == x_single ? y_single : -Inf
        end

        if return_threshold
            return piecewise_interp1, threshold_x
        else
            return piecewise_interp1
        end
    end

    # Step 2: Determine the last valid index before -Inf
    max_valid_index = maximum(valid_indices)  # Last valid index before -Inf
    threshold_x = x[max_valid_index]  # The threshold where y jumps to -Inf

    # Step 3: Extract valid data points
    x_valid = x[valid_indices]  # Feasible x values
    y_valid = y[valid_indices]  # Feasible y values

    # Step 4: Create interpolation function based on `spline` argument
    if spline && length(x_valid) >= 4
        # Use cubic splines only if we have at least 4 valid points
        itp = Spline1D(x_valid, y_valid, k=3)
    else
        # Fallback to linear interpolation if spline isn't possible
        itp = interpolate((x_valid,), y_valid, Gridded(Linear()))
        itp = extrapolate(itp, Interpolations.Flat())  # Avoids artificial smoothing
    end

    # Step 5: Define the final piecewise function
    function piecewise_interp(x_query)
        if x_query > threshold_x  # Beyond valid threshold
            return -Inf
        else
            return itp(x_query)  # Interpolated value
        end
    end

    # Step 6: Return based on `return_threshold` flag
    if return_threshold
        return piecewise_interp, threshold_x
    else
        return piecewise_interp
    end
end

###############################################################################
    ######## 3. INTERPOLATE OPTIMAL LABOR, CONSUMPTION AND UTILITY ########
                    ######## FOR GIVEN (ρ, a, a') #########                    
###############################################################################

function interp_opt_funs(a_grid, opt_c_FOC, opt_l_FOC, gpar, hhpar)
    # Create empty arrays to collect interpolating functions
    opt_c_itp = Array{Any}(undef, gpar.N_rho, gpar.N_a)
    opt_l_itp = Array{Any}(undef, gpar.N_rho, gpar.N_a)
    opt_u_itp = Array{Any}(undef, gpar.N_rho, gpar.N_a)
    # Create empty array to collect max choice a' implying non-negative consumption
    # Used as boundary in the maximisation
    max_a_prime = zeros(gpar.N_rho, gpar.N_a)

    # Interpolate for each possible state (ρ, a) - TBM Deparallelised for safer upper multi-threading
    @inbounds for a_i in 1:gpar.N_a
        for rho_i in 1:gpar.N_rho            
            # Linear interpolation for consumption - Store also max a_prime
            opt_c_itp[rho_i, a_i], max_a_prime[rho_i, a_i] = piecewise_1D_interpolation(a_grid, opt_c_FOC[rho_i, a_i, :], spline=false, return_threshold=true)

            # Linear interpolation for labor
            opt_l_itp[rho_i, a_i] = piecewise_1D_interpolation(a_grid, opt_l_FOC[rho_i, a_i, :], spline=false, return_threshold=false)

            # Linear interpolation for utility
            opt_u_itp[rho_i, a_i] = (a_prime) -> get_utility_hh(opt_c_itp[rho_i, a_i](a_prime), opt_l_itp[rho_i, a_i](a_prime), hhpar)
        end
    end

    return opt_c_itp, opt_l_itp, opt_u_itp, max_a_prime
end


###############################################################################
  ######## 4. INTERPOLATE CONTINUATION VALUE FOR GIVEN STATE (ρ, a) #########
###############################################################################

function interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)
    # This function interpolates the continuation value in the Value Function
    # for each given state in the next period, given today's state
    # cont[ρ, j] = Σ_{ρ'} π(ρ,ρ') V_guess(ρ', j)
    cont = pi_rho * V_guess   # (N_rho, N_a)
    itp_cont = extrapolate(interpolate((rho_grid, a_grid), cont, Gridded(Linear())), Interpolations.Flat())
    # itp_cont = Spline2D(rho_grid, a_grid, cont) #Dierckx spline - ~50 times slower than linear
    itp_cont_wrap = (rho, a_prime) -> itp_cont(rho, a_prime) 
    
    return itp_cont, itp_cont_wrap
end

# # Plot - TBM
# # Select a specific rho index (for example, rho_index = 1, or choose a value of rho)
# rho_index = 7 # You can adjust this index or directly select a value from rho_grid
# rho_val = rho_grid[rho_index]  # Get the specific rho value

# # Get the continuation value and the interpolation function
# itp_cont, cont_interp = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)

# # Now generate values for assets today (a) and tomorrow (a_prime)
# a_values = a_grid  # Asset grid for today
# a_prime_values = a_grid  # Asset grid for tomorrow (same grid, or you can define a new one)

# # Get the interpolated continuation value for these combinations of today and tomorrow's assets
# cont_values = [cont_interp(a_prime, rho_val) for a_prime in a_prime_values]

# # Plot the relationship between assets today (a) vs. assets tomorrow (a') and the continuation value
# Plots.plot(a_values, cont_values, label="Continuation Value", xlabel="Assets Today (a)", ylabel="Continuation Value", title="Interpolation of Continuation Value for ρ = $rho_val")

###############################################################################
  ##################### 5. INTERPOLATE POLICY FUNCTIONS #####################
###############################################################################

function Spline2D_adj(rho_grid, a_grid, matrix2d)
    # Get Dierckx Spline2D
    itp = Spline2D(rho_grid, a_grid, matrix2d)
    # Fix upper and lower bound
    min_val = minimum(matrix2d)
    max_val = maximum(matrix2d)
    return (ρ, a) -> clamp(itp(ρ, a), min_val, max_val)
end


function interpolate_policy_funs(valuef, policy_a, policy_c, policy_l, rho_grid, a_grid)
    # Interpolate and return value function and policy functions
    valuef_int = Spline2D_adj(rho_grid, a_grid, valuef)
    policy_a_int = Spline2D_adj(rho_grid, a_grid, policy_a)
    policy_c_int = Spline2D_adj(rho_grid, a_grid, policy_c)
    policy_l_int = Spline2D_adj(rho_grid, a_grid, policy_l)
    
    return valuef_int, policy_a_int, policy_c_int, policy_l_int
end

