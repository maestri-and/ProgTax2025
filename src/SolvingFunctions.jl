###############################################################################
############################# SOLVINGFUNCTIONS.JL #############################

############### This script defines the main functions to solve ###############
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
include("Interpolants.jl")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 1. TAXES AND UTILITY #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


function compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, net_r, taxes; 
                                          replace_neg_consumption = false, return_cplustax = false)
    """
    Computes household consumption, labor income taxes, and consumption taxes over the grid.

    Args:
        a_grid   : Grid of asset values.
        rho_grid : Grid of productivity levels.
        l_grid   : Grid of labor supply choices.
        gpar     : Struct containing grid parameters.
        w        : Wage rate.
        net_r    : Net Interest rate [(1 - taxes.tau_k)r].
        taxes    : Struct containing tax parameters.
        replace_neg_consumption : Whether to replace negative consumption with -Inf.

    Returns:
        (T_y, hh_consumption, hh_consumption_tax)
    """

    # SECTION 1 - COMPUTE DISPOSABLE INCOME (AFTER WAGE TAX & ASSET RETURNS) #
    
    #Compute gross labor income for each combination of labor and productivity
    y = (l_grid * rho_grid') .* w 

    # Compute labor income taxes for one degree of labor income tax progressivity - Allowing negative tax
    T_y = y .- taxes.lambda_y .* y .^ (1 - taxes.tau_y);

    # Correct for negative tax
    # T_y = max.(T_y, 0)

    # Compute net labor income
    y .-= T_y

    # Compute disposable income after asset transfers (gross capital returns (1+r)a)
    # Disposable income for each possible asset-state-specific interests yielded from t-1 
    # 3rd dim: a
    old_dims_y = ndims(y)
    y = ExpandMatrix(y, gpar.N_a)
    gross_capital_returns = Vector2NDimMatrix((1 + net_r) .* a_grid, old_dims_y)

    y .+= gross_capital_returns;

    ########## SECTION 2 - COMPUTE CONSUMPTION AND CONSUMPTION TAXES ##########

    # Find resource allocated to consumption (consumption + consumption taxes) for each combination of 
    # labor, productivity, assets today
    # 4th dim: a_prime 
    old_dims_y = ndims(y)
    hh_consumption_plus_tax = ExpandMatrix(y, gpar.N_a)
    savings = Vector2NDimMatrix(a_grid, old_dims_y) #a'

    hh_consumption_plus_tax .-= savings;

    # Disentangle consumption from consumption + consumption taxes (Feldstein specification)
    # for one degree of consumption tax progressivity

    # Initialise consumption matrix
    hh_consumption = similar(hh_consumption_plus_tax);

    # Find consumption level
    # ALLOWING FOR CONSUMPTION SUBSIDIES THROUGH CONSUMPTION TAX 
    # Comment the "; notax_upper = break_even" line to allow for redistributive subsidies
    # Through consumption tax

    # Set progressivity rate
    prog_rate = taxes.tau_c

    # Find break-even point 
    break_even = taxes.lambda_c^(1/prog_rate)

    # Allowing for negative tax to better fit interpolant
    @views hh_consumption[:, :, :, :] .= find_c_feldstein.(hh_consumption_plus_tax[:, :, :, :], taxes.lambda_c, prog_rate;
    notax_upper = 0 # break_even - # use to leave negative consumption
    )

    if return_cplustax 
        # Create new array for taxes, so to return the three of them 
        hh_consumption_tax = hh_consumption_plus_tax .- hh_consumption;

        # Correct negative consumption 
        if replace_neg_consumption == true
            @views hh_consumption[hh_consumption .< 0] .= -Inf
        end

        return T_y, hh_consumption, hh_consumption_tax, hh_consumption_plus_tax

    else
        # Retrieve consumption tax - in-place to avoid costly memory allocation
        hh_consumption_plus_tax .-= hh_consumption;
        hh_consumption_tax = hh_consumption_plus_tax

        # Correct negative consumption 
        if replace_neg_consumption == true
            @views hh_consumption[hh_consumption .< 0] .= -Inf
        end

        return T_y, hh_consumption, hh_consumption_tax
    end
end

function compute_utility_grid(hh_consumption, l_grid, hhpar; minus_inf = true,
                              parallelise = true)
    """
    Computes household utility for given consumption and labor choices.

    Args:
        hh_consumption : Precomputed consumption matrix.
        l_grid        : Grid of labor supply choices.
        hhpar : Struct containing household parameters.
        minus_inf     : Whether to replace -Inf with a finite minimum.

    Returns:
        hh_utility : Matrix containing computed household utility.
    """
    ########## SECTION 3 - COMPUTE HOUSEHOLD UTILITY ##########

    # Compute households utility
    hh_utility = similar(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    if parallelise
        @threads for l in 1:gpar.N_l        
            @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                    get_utility_hh.(hh_consumption[l, :, :, :],
                                                    l_grid[l], hhpar), 
                                                    -Inf)
        end
    else
        for l in 1:gpar.N_l        
            @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                    get_utility_hh.(hh_consumption[l, :, :, :],
                                                    l_grid[l], hhpar), 
                                                    -Inf)
        end
    end

    if !minus_inf 
        hh_utility[hh_utility .== -Inf] .= minimum(hh_utility[hh_utility .!= -Inf]) - 1
    end
    return hh_utility
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------# 3. OPTIMAL CONSUMPTION AND LABOR #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# This function uses the budget constraint and the labor FOC 
# derived analytically to estimate optimal consumption and labor as 
# functions of (rho, a, a'), to reduce households problem to a single choice

function find_opt_cons_labor(rho_grid, a_grid, w, net_r, taxes, hhpar, gpar; replace_neg_consumption = true)
    # Create matrices for optimal consumption and optimal labor
    opt_consumption = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)
    opt_labor = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)

    #Pre-allocate vars
    rho = 0
    rhs = 0 
    opt_c = 0
    opt_l = 0
    # Compute upper bound for root finding
    # root_max = maximum(a_grid)*2

    for rho_i in 1:gpar.N_rho
        # Get rho from the grid
        rho = rho_grid[rho_i]
    
        # Define the wage as a function of c using the labor supply function
        wage_star(c) = rho * w * get_opt_labor_from_FOC(c, rho, w, taxes, hhpar)
    
        for a_i in 1:gpar.N_a
            for a_prime_i in 1:gpar.N_a
                # Compute saving returns + saving expenditure (rhs)
                rhs = (1 + net_r) * a_grid[a_i] - a_grid[a_prime_i]
    
                # Define the objective function to solve for optimal consumption (c)
                f = c -> 2 * c - taxes.lambda_c * c^(1 - taxes.tau_c) - taxes.lambda_y * wage_star(c) ^ (1 - taxes.tau_y) - rhs

                # plot_f(f, x_min = 0, x_max = 0.05)

                try
                    # Find solution, if any - Dismissed Newton method as it was giving wrong solution!
                    # Bisection is 5x slower but safer, Brent is slightly faster (4x slower than Secant Order1())                       
                    # opt_c = find_zero(f, 1, Roots.Order1()) 
                    # opt_c = find_zero(f, (1e-6, 0.5), Roots.Bisection())
                    # opt_c = find_zero(f, (0.0, 10), Roots.Brent())
                    
                    # Update: hybrid function trying secant method first and bisection in case of failure
                    opt_c = robust_root_FOC(f, 1e-8, 0.5)   # Use a very low lower bound to avoid bracketing error with bisection

                    # Find optimal labor implied
                    opt_l = get_opt_labor_from_FOC(opt_c, rho, w, taxes, hhpar)

                    # Check whether it is within boundaries, if not 
                    # replace with l = 1 and recompute consumption
                    # exploiting concavity of utility function
                    if opt_l > 1
                        # Assign max to labor
                        opt_l = 1 
                        # Recompute consumption according to budget constraint
                        opt_c = get_opt_c_with_max_labor(rho, a_grid[a_i], a_grid[a_prime_i], w, net_r, taxes; max_labor = 1)
                    end

                    # Check budget constraint holds! - TBM: Consider removing check to improve performances if safe enough
                    if isfinite(opt_c)
                        try 
                            @assert (2*opt_c - taxes.lambda_c*opt_c^(1 - taxes.tau_c)) - (taxes.lambda_y * (rho * w * opt_l) ^ (1 - taxes.tau_y) + rhs) < 0.0001
                        catch AssertionError
                            @info("Budget constraint error at rho_i: $rho_i, a_i: $a_i, a_prime_i: $a_prime_i ")
                            throw(AssertionError)
                        end
                    end

                    # Store values
                    @views opt_consumption[rho_i, a_i, a_prime_i] = opt_c
                    @views opt_labor[rho_i, a_i, a_prime_i] = opt_l

                catch e
                    if isa(e, DomainError) || isa(e, ArgumentError)
                        # Handle DomainError by returning -Inf
                        @views opt_consumption[rho_i, a_i, a_prime_i] = -Inf
                        @views opt_labor[rho_i, a_i, a_prime_i] = Inf
                    else
                        # Rethrow other exceptions
                        @info("Unexpected error at rho_i: $rho_i, a_i: $a_i, a_prime_i: $a_prime_i ")
                        throw(e)
                    end
                end
            end
        end
    end

    # Replace negative consumption points
    if replace_neg_consumption
        # Find all points were the consumption feasibility constraint is broken
        bc_indices = findall(opt_consumption .< 0)
        # Replace consumption matrix with -Inf
        opt_consumption[bc_indices] .= -Inf
        # Replace labor matrix with Inf
        opt_labor[bc_indices] .= Inf
    end
    return opt_consumption, opt_labor
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# 3. VFI #----------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

######################### VFI - EXPLOITING LABOR FOC ##########################

function intVFI_FOC(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hhpar, gpar, comp_params)
    # --- Step 0: Pre-allocate variables ---
    # Pre-allocate value function and policy arrays
    V_guess = zeros(gpar.N_rho, gpar.N_a)
    V_new = similar(V_guess)
    policy_a = similar(V_new)

    # Pre-allocate other variables
    result = 0
    max_error = Inf

    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Interpolate continuation value ---
        itp_cont, itp_cont_wrap = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid);
        
        # --- Step 2: Maximise Bellman operator for a'
        for rho_i in 1:gpar.N_rho
            for a_i in 1:gpar.N_a 
                # Bellman operator, objective function to be maximised
                objective = a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hhpar.beta * itp_cont_wrap(rho_grid[rho_i], a_prime))
    
                # Maximise with respect to a'
                result = optimize(objective, gpar.a_min, max_a_prime[rho_i, a_i], GoldenSection()) 
    
                # Store maximisation results - value     
                @views V_new[rho_i, a_i] =  -Optim.minimum(result)
                
                # Also store the optimal a' for this labor option.
                @views policy_a[rho_i, a_i] = Optim.minimizer(result) 
            end
        end
    
        # --- Step 4: Check convergence ---
        max_error = maximum(abs.(V_new .- V_guess))
        if max_error < comp_params.vfi_tol
            @info("Converged after $iter iterations")
            break
        end
    
        # Otherwise, update the guess.
        # @info("Iteration $iter, error: $max_error")
        V_guess .= V_new
    end
    return V_new, policy_a
end    


function intVFI_FOC_parallel(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hhpar, gpar, comp_params;
    V_guess = zeros(gpar.N_rho, gpar.N_a), parallelise = true)
    """
    Performs Value Function Iteration (VFI) using optimal labor and consumption choices 
    derived from the first-order conditions.

    Args:
        opt_u_itp     : Interpolated utility function {(rho, a) => u(c(a'), l(a'))}
        pi_rho        : Transition matrix for productivity levels
        rho_grid      : Grid of productivity levels
        a_grid        : Grid of asset values
        max_a_prime   : Upper bound for a' choices per (rho, a)
        hhpar : Household parameters (contains beta)
        gpar         : Struct containing grid and problem parameters
        comp_params  : Struct containing VFI computational parameters

    Returns:
        V_new    : Converged value function
        policy_a : Policy function for asset choice a'
    """

    # --- Step 0: Pre-allocate variables ---
    V_guess = V_guess
    V_new = similar(V_guess)
    policy_a = similar(V_guess)

    results = Array{Any}(undef, gpar.N_rho, gpar.N_a)

    # --- Step 1: Begin Value Function Iteration ---
    for iter in 1:comp_params.vfi_max_iter
        # Interpolate continuation value function
        itp_cont, itp_cont_wrap = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)
        
        # --- Step 2: Maximize Bellman equation for each (rho, a) ---
        if parallelise
            @inbounds @threads for a_i in 1:gpar.N_a 
                for rho_i in 1:gpar.N_rho
                    # Define and optimize the objective function
                    results[rho_i, a_i] = optimize(a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hhpar.beta * itp_cont_wrap(rho_grid[rho_i], a_prime)), 
                                                gpar.a_min, max_a_prime[rho_i, a_i], 
                                                GoldenSection()) 

                    # Store results: Value and policy function
                    V_new[rho_i, a_i] = -Optim.minimum(results[rho_i, a_i])
                    policy_a[rho_i, a_i] = Optim.minimizer(results[rho_i, a_i]) 
                end
            end
        else
            @inbounds for a_i in 1:gpar.N_a 
                for rho_i in 1:gpar.N_rho
                    # Define and optimize the objective function
                    results[rho_i, a_i] = optimize(a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hhpar.beta * itp_cont_wrap(rho_grid[rho_i], a_prime)), 
                                                gpar.a_min, max_a_prime[rho_i, a_i], 
                                                GoldenSection()) 

                    # Store results: Value and policy function
                    V_new[rho_i, a_i] = -Optim.minimum(results[rho_i, a_i])
                    policy_a[rho_i, a_i] = Optim.minimizer(results[rho_i, a_i]) 
                end
            end
        end

        # --- Step 3: Check for Convergence ---
        max_error = maximum(abs.(V_new .- V_guess))
        # @info("Iteration $iter, error: $max_error")

        if max_error < comp_params.vfi_tol
            # @info("VFI converged after $iter iterations")
            break
        end

        # Update guess for next iteration
        V_guess .= V_new
    end
    
    return V_new, policy_a
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------------# 4. POLICY FUNCTIONS #----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function compute_policy_matrix(opt_policy_itp, policy_a_int, a_grid, rho_grid)
    """
    Computes a policy function matrix (labor or consumption) based on the optimal asset policy.

    Args:
        opt_policy_itp : Array of 1D interpolations mapping a' → policy value (labor or consumption).
        policy_a_int   : Function or interpolation mapping (rho, a) → a' (optimal asset policy).
        a_grid         : Vector of asset grid values.
        rho_grid       : Vector of productivity grid values.

    Returns:
        A matrix of policy values for l(rho, a) or c(rho, a), depending on the input.
    """

    # Preallocate policy matrix
    policy_matrix = zeros(length(rho_grid), length(a_grid))

    # Compute policy values at optimal a'
    for rho_i in 1:length(rho_grid)
        for a_i in 1:length(a_grid)
            a_prime_opt = policy_a_int(rho_grid[rho_i], a_grid[a_i])  # Get optimal a'
            policy_matrix[rho_i, a_i] = opt_policy_itp[rho_i, a_i](round(a_prime_opt, digits=9))  # Get policy value - round to avoid issues with small differences due to piecewise interpolation
        end
    end

    return policy_matrix
end



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 5. SOLVING HOUSEHOLD PROBLEM #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function SolveHouseholdProblem(a_grid, rho_grid, l_grid, gpar, w, r, 
                               taxes, hhpar, pi_rho, comp_params; 
                               V_guess = V_guess, parallelise = false)
    """
    Solves the household problem via value function iteration using first-order conditions (FOC).

    Returns:
        - hh_labor_tax       : Net labor taxes y - T_y
        - hh_consumption     : Optimal consumption at each (rho, a, a')
        - hh_consumption_tax : Consumption tax paid
        - opt_c_FOC          : Optimal consumption from FOC
        - opt_l_FOC          : Optimal labor from FOC
        - valuef             : Value function at convergence
        - policy_a           : Policy function for a'
        - policy_l           : Labor policy function l(rho, a)
        - policy_c           : Consumption policy function c(rho, a)
    """
    ########## PRELIMINARY ##########
    # Allocate net interest rate (for simplicity)
    net_r = (1 - taxes.tau_k)r

    ########## SECTION 1 - COMPUTE TAXES, CONSUMPTION AND UTILITY ##########

    # @info("Solving budget constraint...")

    hh_labor_tax, hh_consumption, hh_consumption_tax = compute_consumption_grid_for_itp(
        a_grid, rho_grid, l_grid, gpar, w, r, net_r, taxes;
        replace_neg_consumption = true
    );

    ########## SECTION 2 - COMPUTE OPTIMAL CONSUMPTION AND LABOR ##########

    # @info("Pinning down optimal labor and consumption using labor FOC...")

    opt_c_FOC, opt_l_FOC = find_opt_cons_labor(
        rho_grid, a_grid, w, net_r, taxes, hhpar, gpar;
        replace_neg_consumption = true
    );

    ########## SECTION 3 - INTERPOLATE POLICIES FROM FIRST ORDER CONDITIONS ##########

    opt_c_itp, opt_l_itp, opt_u_itp, max_a_prime = interp_opt_funs(
        a_grid, opt_c_FOC, opt_l_FOC, gpar, hhpar
    );

    ########## SECTION 4 - SOLVE VALUE FUNCTION ITERATION ##########

    # @info("Launching VFI...")

    @elapsed valuef, policy_a = intVFI_FOC_parallel(
        opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hhpar, gpar, comp_params;
        V_guess = V_guess, parallelise = parallelise
    )

    ########## SECTION 5 - RECONSTRUCT FINAL POLICY FUNCTIONS ##########

    policy_a_int = Spline2D_adj(rho_grid, a_grid, policy_a)

    policy_l = compute_policy_matrix(opt_l_itp, policy_a_int, a_grid, rho_grid)
    policy_c = compute_policy_matrix(opt_c_itp, policy_a_int, a_grid, rho_grid)

    ########## SECTION 6 - RUN CHECKS ON OUTPUT ##########
    # Check that value and policy functions contain only finite values
    for m in [valuef, policy_a, policy_c, policy_l]
        if all(isfinite.(m))
            nothing 
        else 
            @error("Issue with r=$r, w=$w, taxes=$(taxes)! \n Non-finite values in value or policy functions! Check VFI!")
            error("Non-finite values in value or policy functions! Check VFI!")
        end
    end
    
    return hh_labor_tax, hh_consumption, hh_consumption_tax, opt_c_FOC, opt_l_FOC, valuef, policy_a, policy_l, policy_c
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------------# 6. COMPUTING AGGREGATES #-------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function stationary_distribution(a_grid, pi_rho, policy_a, gpar; tol=1e-9, max_iter=100_000)    
    # Initial guess: uniform distribution
    dist = fill(1.0 / (gpar.N_a * gpar.N_rho), gpar.N_rho, gpar.N_a)
    dist_new = similar(dist)

    for iter in 1:max_iter
        fill!(dist_new, 0.0)

        for a_i in 1:gpar.N_a
            for rho_i in 1:gpar.N_rho
                a_prime = policy_a[rho_i, a_i]  # next asset level (not necessarily on grid)

                # Find interpolation weights
                if a_prime <= a_grid[1]
                    a_i_low, a_i_high, w_high = 1, 1, 1.0
                elseif a_prime >= a_grid[end]
                    a_i_low, a_i_high, w_high = gpar.N_a, gpar.N_a, 1.0
                else
                    a_i_high = searchsortedfirst(a_grid, a_prime)
                    a_i_low = a_i_high - 1
                    w_high = (a_prime - a_grid[a_i_low]) / (a_grid[a_i_high] - a_grid[a_i_low])
                end

                # Transition over productivity
                for rho_i_prime in 1:gpar.N_rho
                    prob = pi_rho[rho_i, rho_i_prime]
                    mass = dist[rho_i, a_i] * prob
                    dist_new[rho_i_prime, a_i_low] += (1 - w_high) * mass
                    dist_new[rho_i_prime, a_i_high] += w_high * mass
                end
            end
        end

        if maximum(abs.(dist_new .- dist)) < tol
            # @info("Stationary Distribution: Converged after $iter iterations")
            return dist_new
        end
        dist .= dist_new
    end

    @error("Stationary distribution did not converge")
    error("Stationary distribution did not converge")

    return dist_new
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 7. SOLVING GENERAL EQUILIBRIUM #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Solving model using bisection 

function ComputeEquilibrium_Bisection(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params; collect_errors = true, bisection_weight = 0
)
    #--- Initial bounds for interest rate (r) for bisection method
    r_low = -fpar.delta
    r_high = 0.1 # 1 / hhpar.beta - 1 # is too low!!!
    r_mid = (r_low + r_high) / 2
    bw = bisection_weight  # Weight for dampened update of r

    #--- Wage implied by firm's FOC given r
    w = cd_implied_opt_wage(r_mid)

    if collect_errors
            #--- Store r points and K_error(r) to study functional form
            rates = []
            errors = [] 
    end

    #--- Initialise guess for value function
    V_guess = zeros(gpar.N_rho, gpar.N_a)

    #--- Begin equilibrium loop
    for iter in 1:comp_params.ms_max_iter

        ###### 1. Household Problem ######
        (hh_labor_tax, hh_consumption, hh_consumption_tax,
         opt_c_FOC, opt_l_FOC, valuef, policy_a,
         policy_l, policy_c) = SolveHouseholdProblem(
             a_grid, rho_grid, l_grid, gpar, w, r_mid, taxes,
             hhpar, pi_rho, comp_params;
             V_guess = zeros(gpar.N_rho, gpar.N_a)
         )
        
        ###### 1bis. Store value function for new guess
        # V_guess .= valuef

        ###### 2. Stationary Distribution ######
        stat_dist = stationary_distribution(
            a_grid, pi_rho, policy_a, gpar;
            tol = 1e-10, max_iter = 100_000
        )

        ###### 3. Aggregates ######
        asset_supply = sum(stat_dist * a_grid)  # asset by productivity
        effective_labor_supply = sum(stat_dist .* policy_l .* rho_grid)

        ###### 4. Firm's Capital Demand from FOC ######
        asset_demand = ((fpar.alpha * fpar.tfp) / (r_mid + fpar.delta)) ^
                       (1 / (1 - fpar.alpha)) * effective_labor_supply

        ###### 5. Check for Market Clearing ######
        K_error = asset_demand - asset_supply

        @info("Iter $iter: r = $(round(r_mid, digits=5)), w = $(round(w, digits=5)), K_supply = $(round(asset_supply, digits=5)), K_demand = $(round(asset_demand, digits=5)), error = $(round(K_error, digits=5))")
    
        if collect_errors
            #--- Store r points and K_error(r) to study functional form
            push!(rates, r_mid)
            push!(errors, K_error) 
        end

        if abs(K_error) < comp_params.ms_tol
            @info("✅ Equilibrium found: r = $r_mid, w = $w after $iter iterations")
            if collect_errors
                return r_mid, w, stat_dist, valuef, policy_a, policy_l, policy_c, rates, errors
            else
                return r_mid, w, stat_dist, valuef, policy_a, policy_l, policy_c
            end
        end

        ###### 6. Bisection Update of Interest Rate ######
        if K_error > 0
            r_low = r_mid  # Excess demand → raise r
        else
            r_high = r_mid  # Excess supply → lower r
        end

        r_new = bw * r_mid + (1 - bw) * ((r_low + r_high) / 2)
        r_mid = r_new
        w = cd_implied_opt_wage(r_mid)
    end

    @error("❌ Equilibrium not found within max iterations.")
    error("❌ Equilibrium not found within max iterations.")
end

function ComputeEquilibrium_Newton(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params; collect_errors = true, damping_weight = 1,
    prevent_Newton_jump = false, initial_r = nothing,
    print_info = true, parallelise = true
)
    #--- Initial guess for interest rate
    if isnothing(initial_r)
        r_mid = 0.025
    else
        r_mid = initial_r
    end
    bw = damping_weight  # Damping parameter

    #--- Wage implied by firm's FOC given r
    w = cd_implied_opt_wage(r_mid)

    if collect_errors
        rates = []
        errors = []
    end

    #--- Initialise guess for value function
    V_guess = zeros(gpar.N_rho, gpar.N_a)

    for iter in 1:comp_params.ms_max_iter

        ###### 1. Household Problem ######
        (_, _, _, _, _, valuef, policy_a, policy_l, policy_c) = SolveHouseholdProblem(
            a_grid, rho_grid, l_grid, gpar, w, r_mid, taxes,
            hhpar, pi_rho, comp_params; 
            V_guess = zeros(gpar.N_rho, gpar.N_a), 
            parallelise = parallelise
        )

        ###### 1bis. Store value function for new guess
        # V_guess .= valuef

        ###### 2. Stationary Distribution ######
        stat_dist = stationary_distribution(
            a_grid, pi_rho, policy_a, gpar;
            tol = 1e-10, max_iter = 100_000
        )

        ###### 3. Aggregates ######
        asset_supply = sum(stat_dist * a_grid)
        effective_labor_supply = sum(stat_dist .* policy_l .* rho_grid)

        ###### 4. Firm's Capital Demand ######
        asset_demand = ((fpar.alpha * fpar.tfp) / (r_mid + fpar.delta)) ^
                       (1 / (1 - fpar.alpha)) * effective_labor_supply

        ###### 5. Market Clearing Error ######
        K_error = asset_demand - asset_supply

        if print_info
            @info("Iter $iter: r = $(round(r_mid, digits=6)), K_supply = $(round(asset_supply, digits=6)), K_demand = $(round(asset_demand, digits=6)), error = $(round(K_error, digits=8))")
        end

        if collect_errors
            push!(rates, r_mid)
            push!(errors, K_error)
        end

        if abs(K_error) < comp_params.ms_tol
            if print_info
                @info("✅ Equilibrium found: r = $r_mid, w = $w after $iter iterations")
            end
            if collect_errors
                return r_mid, w, stat_dist, valuef, policy_a, policy_l, policy_c, rates, errors
            else
                return r_mid, w, stat_dist, valuef, policy_a, policy_l, policy_c
            end
        end

        ###### 6. Finite Difference Derivative ######
        dr = 1e-5
        r_up = r_mid + dr
        w_up = cd_implied_opt_wage(r_up)

        (_, _, _, _, _, _, policy_a_up, policy_l_up, _) = SolveHouseholdProblem(
            a_grid, rho_grid, l_grid, gpar, w_up, r_up, taxes,
            hhpar, pi_rho, comp_params;
            V_guess = V_guess, 
            parallelise = parallelise
        )

        stat_dist_up = stationary_distribution(
            a_grid, pi_rho, policy_a_up, gpar;
            tol = 1e-10, max_iter = 100_000
        )

        asset_supply_up = sum(stat_dist_up * a_grid)
        effective_labor_supply_up = sum(stat_dist_up .* policy_l_up .* rho_grid)
        asset_demand_up = ((fpar.alpha * fpar.tfp) / (r_up + fpar.delta)) ^
                          (1 / (1 - fpar.alpha)) * effective_labor_supply_up
        K_error_up = asset_demand_up - asset_supply_up

        K_derivative = (K_error_up - K_error) / dr

        ###### 7. Newton Update ######
        if abs(K_derivative) < 1e-8
            @warn("⚠️ Derivative near zero — using damped fallback")
            r_new = r_mid - 0.1 * K_error
        else
            r_new = r_mid - K_error / K_derivative
        end

        # Damping for stability
        r_mid = (1 - bw) * r_mid + bw * r_new

        # Enforce reasonable r range - TBM 
        if prevent_Newton_jump
            r_mid = clamp(r_new, 0.00001, 0.06)
        end
        
        w = cd_implied_opt_wage(r_mid)

    end

    @error("❌ Equilibrium not found within max iterations.")
    error("❌ Equilibrium not found within max iterations.")
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------# 8. COMPUTING AGGREGATE DISTRIBUTIONS #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function compute_aggregates_and_check(stat_dist, policy_a, policy_c, policy_l, rho_grid, a_grid, w, r, taxes; 
                                      raise_bc_error = true, raise_clearing_error = true)
    ###### 1. Compute aggregates ######
    # Consumption 
    distC = stat_dist .* policy_c
    aggC = sum(stat_dist .* policy_c)

    # Capital
    distK = policy_a .* stat_dist
    aggK = sum(distK)

    # Labor
    distH = policy_l .* stat_dist
    aggH = sum(distH)

    # Effective labor
    distL = distH .* rho_grid
    aggL = sum(distL)

    # Aggregate output
    aggY = cd_production(fpar.tfp, fpar.alpha, aggK, aggL)

    # Consumption Tax
    consumption_tax_policy = policy_c .- taxes.lambda_c .* policy_c .^ (1 - taxes.tau_c) 
    distCtax = stat_dist .* consumption_tax_policy 
    aggT_c = sum(distCtax)              # Net consumption tax revenue, after redistribution if any

    # Labor Tax
    policy_l_incpt = policy_l .* rho_grid .* w    # diag(rho_grid)*policy_l*w
    labor_tax_policy = policy_l_incpt .- taxes.lambda_y .* policy_l_incpt .^ (1 - taxes.tau_y)
    distWtax = stat_dist .* labor_tax_policy
    aggT_y = sum(distWtax)                    # Net labor tax revenue, after redistribution if any

    # Capital Tax
    distKtax = (taxes.tau_k * r) .* a_grid
    aggT_k = sum(stat_dist * distKtax)

    # Government expenditure
    aggG = aggT_y + aggT_c + aggT_k

    ###### 2. Check market clearing and budget constraint ######

    # Doublecheck goods' market clearing
    # Y = C + I + G
    excess_prod = aggY - (aggC + fpar.delta * aggK + aggG)
    if abs(excess_prod) > 0.01 && raise_clearing_error
        @error("Market for goods did not clear!")
        error("Market for goods did not clear!")
    elseif abs(excess_prod) > 0.01 && !raise_clearing_error
        @warn("Large residual for the goods' market ($excess_prod)! Doublecheck accuracy!")
    end

    # Doublecheck budget constraint holding for optimal policies - Get max discrepancy
    bc_max_discrepancy = findmax(abs.(policy_c .+ consumption_tax_policy .- (policy_l_incpt .- labor_tax_policy .+ ((1 + (1 - taxes.tau_k)r) .* (ones(7, 1) * a_grid')) .- policy_a)))
    if bc_max_discrepancy[1] > 0.01 && raise_bc_error
        @error("Max budget constraint discrepancy is larger than 0.01! Doublecheck accuracy! $bc_max_discrepancy")
        error("Max budget constraint discrepancy is larger than 0.01! Doublecheck accuracy! $bc_max_discrepancy")
    elseif bc_max_discrepancy[1] > 0.01 && !raise_bc_error
        @warn("Max budget constraint discrepancy is larger than 0.01! Doublecheck accuracy! $bc_max_discrepancy")
    end

    return(distC, distK, distH, distL,
           distCtax, distWtax, distKtax, 
           aggC, aggK, aggH, aggL, aggG, aggY, 
           aggT_c, aggT_y, aggT_k, 
           excess_prod, bc_max_discrepancy)
end

function compute_government_revenue(stat_dist, policy_c, policy_l, a_grid, rho_grid, r, w, taxes)
    # Consumption Tax
    consumption_tax_policy = policy_c .- taxes.lambda_c .* policy_c .^ (1 - taxes.tau_c) 
    distCtax = stat_dist .* consumption_tax_policy 
    aggT_c = sum(distCtax)              # Net consumption tax revenue, after redistribution if any

    # Labor Tax
    policy_l_incpt = policy_l .* rho_grid .* w    # diag(rho_grid)*policy_l*w
    labor_tax_policy = policy_l_incpt .- taxes.lambda_y .* policy_l_incpt .^ (1 - taxes.tau_y)
    distWtax = stat_dist .* labor_tax_policy
    aggT_y = sum(distWtax)                    # Net labor tax revenue, after redistribution if any

    # Capital Tax
    distKtax = (taxes.tau_k * r) .* a_grid
    aggT_k = sum(stat_dist * distKtax)

    # Government expenditure
    return aggT_y + aggT_c + aggT_k
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 9. SOLVING FOR EQUIVALENT TAX REGIME  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function TwoLevelEquilibriumNewton(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, t_taxes,
    pi_rho, comp_params, G_target;
    adjust_par::Symbol = :tau_c,
    initial_r = 0.03, 
    tol = 1e-6,
    max_iter = 50,
    regime_search_only = false
)   
    # Initialise new tax regime vector (copy base taxes)
    new_taxes = deepcopy(t_taxes)

    # Initial guess for the tax parameter to adjust
    tpar_guess = getproperty(new_taxes, adjust_par)
    h = 1e-4  # step for finite difference


    for iter in 1:max_iter
        # Update the tax parameter
        setproperty!(new_taxes, adjust_par, tpar_guess)

        # === Inner Newton to find r ===
        r_eq, w_eq, stat_dist, valuef, policy_a, policy_l, policy_c = ComputeEquilibrium_Newton(
            a_grid, rho_grid, l_grid, gpar, hhpar, fpar, new_taxes,
            pi_rho, comp_params;
            initial_r = initial_r,
            collect_errors = false,
            print_info = false,
            parallelise = false
        )

        # === Compute government revenue and its gap ===
        aggG = compute_government_revenue(stat_dist, policy_c, policy_l, a_grid, rho_grid, r_eq, w_eq, new_taxes)
        G_error = G_target - aggG

        @info("[Thread $(Threads.threadid())] Outer Newton Iter $iter: $(adjust_par) = $(tpar_guess), G_error = $(G_error)")

        if abs(G_error) < tol
            @info("[Thread $(Threads.threadid())] ✅ Converged: r = $(r_eq), $(adjust_par) = $(tpar_guess)")
            if regime_search_only
                return tpar_guess, r_eq, aggG
            else
                return tpar_guess, r_eq, w_eq, stat_dist, valuef, policy_a, policy_l, policy_c
            end
        end

        # === Finite difference to compute derivative of G w.r.t. tax param ===
        tpar_up = tpar_guess + h
        setproperty!(new_taxes, adjust_par, tpar_up)

        r_up, w_up, stat_dist_up, valuef_up, policy_a_up, policy_l_up, policy_c_up = ComputeEquilibrium_Newton(
            a_grid, rho_grid, l_grid, gpar, hhpar, fpar, new_taxes,
            pi_rho, comp_params;
            initial_r = initial_r,
            collect_errors = false,
            print_info = false,
            parallelise = false
        )

        aggG_up = compute_government_revenue(stat_dist_up, policy_c_up, policy_l_up, a_grid, rho_grid, r_up, w_up, new_taxes)
        G_error_up = G_target - aggG_up

        dG = (G_error_up - G_error) / h

        # Newton update for tax parameter
        tpar_guess -= G_error / dG
    end

    error("❌ Joint Newton (outer on $(adjust_par)) did not converge.")
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------------# 10. ROBUSTNESS CHECKS  #--------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


function compute_tax_components(stat_dist, policy_c, policy_l, 
    a_grid, rho_grid, r, w, taxes)
        # Consumption Tax
    consumption_tax_policy = policy_c .- taxes.lambda_c .* policy_c .^ (1 - taxes.tau_c) 
    distCtax = stat_dist .* consumption_tax_policy 
    aggT_c = sum(distCtax)              # Net consumption tax revenue, after redistribution if any

    # Labor Tax
    policy_l_incpt = policy_l .* rho_grid .* w    # diag(rho_grid)*policy_l*w
    labor_tax_policy = policy_l_incpt .- taxes.lambda_y .* policy_l_incpt .^ (1 - taxes.tau_y)
    distWtax = stat_dist .* labor_tax_policy
    aggT_y = sum(distWtax)                    # Net labor tax revenue, after redistribution if any

    # Capital Tax
    distKtax = (taxes.tau_k * r) .* a_grid
    aggT_k = sum(stat_dist * distKtax)

    return aggT_c, aggT_y, aggT_k
end

function MultiDimEquilibriumNewton(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, base_taxes,
    pi_rho, comp_params, G_target, target_val;
    target = "revenue",
    initial_x = [0.03, base_taxes.tau_c, base_taxes.lambda_c],
    tol = 1e-6,
    max_iter = 50,
    damping_weight = 0.2
)
    # Create mutable copy of taxes
    new_taxes = deepcopy(base_taxes)
    x = copy(initial_x)  # [r, tau_c, lambda_c]
    h = 1e-4

    for iter in 1:max_iter
        r, tau_c, lambda_c = x
        setproperty!(new_taxes, :tau_c, tau_c)
        setproperty!(new_taxes, :lambda_c, lambda_c)

        # === Solve household problem directly at fixed r ===
        w = cd_implied_opt_wage(r)
        _, _, _, _, _, valuef, policy_a, policy_l, policy_c = SolveHouseholdProblem(
            a_grid, rho_grid, l_grid, gpar, w, r, new_taxes,
            hhpar, pi_rho, comp_params,
            V_guess = zeros(gpar.N_rho, gpar.N_a),
            parallelise = false
        )

        stat_dist = stationary_distribution(
            a_grid, pi_rho, policy_a, gpar,
            tol = 1e-10, max_iter = 100_000
        )

        # === Compute residuals ===
        aggK = sum(stat_dist .* a_grid')
        effective_L = sum(stat_dist .* policy_l .* rho_grid)
        K_demand = ((fpar.alpha * fpar.tfp) / (r + fpar.delta)) ^ (1 / (1 - fpar.alpha)) * effective_L
        F1 = K_demand - aggK

        
        Tc, Ty, Tk = compute_tax_components(stat_dist, policy_c, policy_l, a_grid, rho_grid, r, w, new_taxes)
        T = Tc + Ty + Tk
        F2 = G_target - T
        if target == "revenue"
            F3 = target_val - (Tc / T)
        elseif target == "aer"
            # Compute average effective rate 
            consumption_tax_policy = policy_c .- new_taxes.lambda_c .* policy_c .^ (1 - new_taxes.tau_c) 
            cons_tax_eff_rates = consumption_tax_policy ./ policy_c
            aer = sum(cons_tax_eff_rates .* stat_dist)
            # Define error
            F3 = target_val - aer
        else
            error("UnDefError: Check your target!")
        end

        # println("Capital error: $F1, G = $T, Share of Consumption Taxes = $(Tc / T)")

        F = [F1, F2, F3]

        if norm(F) < tol
            println("✅ Converged: r = $(r), tau_c = $(tau_c), lambda_c = $(lambda_c)")
            return x[3], x[2], x[1], w, stat_dist, valuef, policy_a, policy_l, policy_c
        end

        # === Numerical Jacobian ===
        J = zeros(3, 3)
        for i in 1:3
            x_perturb = copy(x)
            x_perturb[i] += h
            r_p, tau_p, lambda_p = x_perturb

            setproperty!(new_taxes, :tau_c, tau_p)
            setproperty!(new_taxes, :lambda_c, lambda_p)

            w_p = cd_implied_opt_wage(r_p)
            _, _, _, _, _, _, policy_a_p, policy_l_p, policy_c_p = SolveHouseholdProblem(
                a_grid, rho_grid, l_grid, gpar, w_p, r_p, new_taxes,
                hhpar, pi_rho, comp_params,
                V_guess = zeros(gpar.N_rho, gpar.N_a),
                parallelise = false
            )

            stat_dist_p = stationary_distribution(
                a_grid, pi_rho, policy_a_p, gpar,
                tol = 1e-10, max_iter = 100_000
            )

            aggK_p = sum(stat_dist_p .* a_grid')
            effective_L_p = sum(stat_dist_p .* policy_l_p .* rho_grid)
            K_demand_p = ((fpar.alpha * fpar.tfp) / (r_p + fpar.delta)) ^ (1 / (1 - fpar.alpha)) * effective_L_p
            F1_p = K_demand_p - aggK_p

            Tc_p, Ty_p, Tk_p = compute_tax_components(stat_dist_p, policy_c_p, policy_l_p, a_grid, rho_grid, r_p, w_p, new_taxes)
            T_p = Tc_p + Ty_p + Tk_p
            F2_p = G_target - T_p

            if target == "revenue"
                F3_p = target_val - (Tc_p / T_p)
            elseif target == "aer"
                # Compute average effective rate 
                consumption_tax_policy_p = policy_c_p .- new_taxes.lambda_c .* policy_c_p .^ (1 - new_taxes.tau_c) 
                cons_tax_eff_rates_p = consumption_tax_policy_p ./ policy_c_p
                aer_p = sum(cons_tax_eff_rates_p .* stat_dist_p)
                # Define error
                F3_p = target_val - aer_p
            else
                error("UnDefError: Check your target!")
            end
            
            J[:, i] .= ([F1_p, F2_p, F3_p] .- F) ./ h
        end

        dx = J \ F
        x -= damping_weight * dx
        # Make sure bounds are satisfied 
        if x[3] != clamp(x[3], 0.0, 1.0)
            @warn("lambda_c hit a bound: $(x[3])")
            x[3] = clamp(x[3], 0.0, 1.0)    # lambda_c ∈ [0, 1]
        end
        

        println("Iter $iter: r = $(x[1]), tau_c = $(x[2]), lambda_c = $(x[3]), norm(F) = $(norm(F))")
    end

    error("❌ Multi-dim Newton did not converge.")
    
end
