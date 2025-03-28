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
include("Households.jl")
include("AuxiliaryFunctions.jl")
include("Interpolants.jl")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 1. TAXES AND UTILITY #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


### Simpler version: single rate of progressivity for consumption taxes and labor income taxes
#### Memory efficient version - More in-place operations + @views
### Memory gains: 24% 

function compute_hh_taxes_consumption_utility_ME(a_grid, rho_grid, l_grid, gpar, w, net_r, taxes, hhpar)

    # SECTION 1 - COMPUTE DISPOSABLE INCOME (AFTER WAGE TAX & ASSET RETURNS) #
    
    #Compute gross labor income for each combination of labor and productivity
    y = (l_grid * rho_grid') .* w 

    # Compute labor income taxes for one degree of labor income tax progressivity - Allowing negative tax
    T_y = y .- taxes.lambda_y .* y .^ (1 - taxes.tau_y);

    # Correct for negative tax
    # T_y = max.(T_y, 0)

    # Compute net labor income
    y .-= T_y

    # Compute disposable income after asset transfers (net capital returns (1+(1 - tau_k)r)a)
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
    hh_consumption = copy(hh_consumption_plus_tax);

    # Find consumption level
    # ALLOWING FOR CONSUMPTION SUBSIDIES THROUGH CONSUMPTION TAX 
    # Comment the "; notax_upper = break_even" line to allow for redistributive subsidies
    # Through consumption tax

    # Set progressivity rate
    prog_rate = taxes.tau_c

    # Find break-even point 
    break_even = taxes.lambda_c^(1/prog_rate)

    # To allow for redistributive subsidies remove the notax_upper argument from the function
    @views hh_consumption[:, :, :, :] .= find_c_feldstein.(hh_consumption_plus_tax[:, :, :, :], taxes.lambda_c, prog_rate
    ; # notax_upper = break_even
    )

    # Retrieve consumption tax - in-place to avoid costly memory allocation
    hh_consumption_plus_tax .-= hh_consumption;

    # Rename for clarity
    hh_consumption_tax = hh_consumption_plus_tax

    # Correct negative consumption 
    @views hh_consumption[hh_consumption .< 0] .= -Inf

    ########## SECTION 3 - COMPUTE HOUSEHOLD UTILITY ##########

    # Compute households utility
    hh_utility = similar(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    @threads for l in 1:gpar.N_l        @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :],
                                                l_grid[l], hhpar), 
                                                hh_consumption[l, :, :, :])
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_utility
end

###############################################################################
##### SPLIT AND WRITE TO DISK TO SAVE MEMORY #####
###############################################################################


function compute_consumption_grid(a_grid, rho_grid, l_grid, gpar, w, net_r, taxes)
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
    hh_consumption = copy(hh_consumption_plus_tax);

    # Find consumption level
    # ALLOWING FOR CONSUMPTION SUBSIDIES THROUGH CONSUMPTION TAX 
    # Comment the "; notax_upper = break_even" line to allow for redistributive subsidies
    # Through consumption tax

    # Set progressivity rate
    prog_rate = taxes.tau_c

    # Find break-even point 
    break_even = taxes.lambda_c^(1/prog_rate)

    # To allow for redistributive subsidies remove the notax_upper argument from the function
    @views hh_consumption[:, :, :, :] .= find_c_feldstein.(hh_consumption_plus_tax[:, :, :, :], taxes.lambda_c, prog_rate
    ; # notax_upper = break_even
    )

    # Retrieve consumption tax - in-place to avoid costly memory allocation
    hh_consumption_plus_tax .-= hh_consumption;

    # Rename for clarity
    hh_consumption_tax = hh_consumption_plus_tax

    # Correct negative consumption 
    @views hh_consumption[hh_consumption .< 0] .= -Inf

    # Write to disk consumption tax matrix for later usage 
    filename = "ConsumptionTax_l$(gpar.N_l)_a$(gpar.N_a).txt"
    SaveMatrix(hh_consumption_tax, "output/temp/" * filename)
    return T_y, hh_consumption
end


function compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, net_r, taxes; replace_neg_consumption = false, return_cplustax = false)
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

function compute_utility_grid(hh_consumption, l_grid, hhpar; minus_inf = true)
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
    @threads for l in 1:gpar.N_l        
        @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :],
                                                l_grid[l], hhpar), 
                                                -Inf)
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
# functions of (ρ, a, a'), to reduce households problem to a single choice

function find_opt_cons_labor(rho_grid, a_grid, w, net_r, taxes, hhpar, gpar; enforce_labor_cap = true, replace_neg_consumption = true)
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
    
        if enforce_labor_cap
            for a_i in 1:gpar.N_a
                for a_prime_i in 1:gpar.N_a
                    # Compute saving returns + saving expenditure (rhs)
                    rhs = (1 + net_r) * a_grid[a_i] - a_grid[a_prime_i]
        
                    # Define the objective function to solve for optimal consumption (c)
                    f = c -> 2 * c - taxes.lambda_c * c^(1 - taxes.tau_c) - taxes.lambda_y * wage_star(c) ^ (1 - taxes.tau_y) - rhs

                    # plot_f(f, x_min = 0, x_max = 0.5)

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
                                println("Budget constraint error at rho_i: $rho_i, a_i: $a_i, a_prime_i: $a_prime_i ")
                                throw(AssertionError)
                            end
                        end

                        # Store values
                        @views opt_consumption[rho_i, a_i, a_prime_i] = opt_c
                        @views opt_labor[rho_i, a_i, a_prime_i] = opt_l

                    catch e
                        if isa(e, DomainError)
                            # Handle DomainError by returning -Inf
                            @views opt_consumption[rho_i, a_i, a_prime_i] = -Inf
                            @views opt_labor[rho_i, a_i, a_prime_i] = Inf
                        else
                            # Rethrow other exceptions
                            println("Unexpected error at rho_i: $rho_i, a_i: $a_i, a_prime_i: $a_prime_i ")
                            throw(e)
                        end
                    end
                end
            end
        else
            for a_i in 1:gpar.N_a
                for a_prime_i in 1:gpar.N_a
                    # Compute saving returns + saving expenditure (rhs)
                    rhs = (1 + net_r) * a_grid[a_i] - a_grid[a_prime_i]
        
                    # Define the objective function to solve for optimal consumption (c)
                    f = c -> 2 * c - taxes.lambda_c * c^(1 - taxes.tau_c) - taxes.lambda_y * wage_star(c) ^ (1 - taxes.tau_y) - rhs
        
                    try
                        # Find solution, if any - Dismissed Newton method as it was giving wrong solution!
                        # Bisection is 5x slower but safer, Brent is slightly faster (4x slower than Secant Order1())
                        # opt_c = find_zero(f, 0.5, Roots.Order1()) #0.5 Initial guess, adjustable
                        # opt_c = find_zero(f, (1e-6, 2*a_grid_max), Roots.Bisection())
                        opt_c = find_zero(f, (1e-6, root_max), Roots.Brent())

                        # Find optimal labor implied
                        opt_l = get_opt_labor_from_FOC(opt_c, rho, w, taxes, hhpar)

                        

                        @views opt_consumption[rho_i, a_i, a_prime_i] = opt_c 
                        # Get optimal labor
                        @views opt_labor[rho_i, a_i, a_prime_i] = opt_l
                    catch e
                        if isa(e, DomainError)
                            # Handle DomainError by returning -Inf
                            @views opt_consumption[rho_i, a_i, a_prime_i] = -Inf
                            @views opt_labor[rho_i, a_i, a_prime_i] = -Inf
                        else
                            # Rethrow other exceptions
                            throw(e)
                        end
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
            println("Converged after $iter iterations")
            break
        end
    
        # Otherwise, update the guess.
        # println("Iteration $iter, error: $max_error")
        V_guess .= V_new
    end
    return V_new, policy_a
end    


function intVFI_FOC_parallel(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hhpar, gpar, comp_params)
    """
    Performs Value Function Iteration (VFI) using optimal labor and consumption choices 
    derived from the first-order conditions.

    Args:
        opt_u_itp     : Interpolated utility function {(ρ, a) => u(c(a'), l(a'))}
        pi_rho        : Transition matrix for productivity levels
        rho_grid      : Grid of productivity levels
        a_grid        : Grid of asset values
        max_a_prime   : Upper bound for a' choices per (ρ, a)
        hhpar : Household parameters (contains β)
        gpar         : Struct containing grid and problem parameters
        comp_params  : Struct containing VFI computational parameters

    Returns:
        V_new    : Converged value function
        policy_a : Policy function for asset choice a'
    """

    # --- Step 0: Pre-allocate variables ---
    V_guess = zeros(gpar.N_rho, gpar.N_a)
    V_new = similar(V_guess)
    policy_a = similar(V_guess)

    results = Array{Any}(undef, gpar.N_rho, gpar.N_a)

    # --- Step 1: Begin Value Function Iteration ---
    for iter in 1:comp_params.vfi_max_iter
        # Interpolate continuation value function
        itp_cont, itp_cont_wrap = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)
        
        # --- Step 2: Maximize Bellman equation for each (ρ, a) ---
        @inbounds @threads for a_i in 1:gpar.N_a 
            for rho_i in 1:gpar.N_rho
                # # Solving VFI issues - plotting 
                # # Compute the objective function values
                # # Plot only utility
                # plot_utility_vs_aprime(rho_i, a_i, opt_u_itp, opt_c_itp, opt_l_itp, a_grid, max_a_prime, hhpar)

                # obj_values = [
                #     (opt_u_itp[rho_i, a_i](a_p))
                #     for a_p in a_prime_values
                # ]
                # # Plot 
                # plot(a_prime_values, obj_values, title="Utility for ρ=$(rho_grid[rho_i]), a=$(a_grid[a_i])",
                #      xlabel="Future Assets (a')", ylabel="Utility", lw=2, legend=false)


                # # Plot only discounted continuation value
                # plot_interpolation_vs_knots(V_guess, pi_rho, rho_grid, a_grid, rho_level = 7)

                # obj_values = [
                #     (itp_cont_wrap(rho_grid[rho_i], a_p))
                #     for a_p in a_prime_values
                # ]

                # # Plot the objective function
                # plot(a_prime_values, obj_values, title="Continuation value for ρ=$(rho_grid[rho_i]), a=$(a_grid[a_i])",
                #      xlabel="Future Assets (a')", ylabel="Discounted continuation Value", lw=2, legend=false)

                # # Plot objective
                # obj_values = [
                #     (opt_u_itp[rho_i, a_i](a_p) + hhpar.beta * itp_cont_wrap(rho_grid[rho_i], a_p))
                #     for a_p in a_prime_values
                # ]

                # # Plot the objective function
                # plot(a_prime_values, obj_values, title="Objective Function for ρ=$(rho_grid[rho_i]), a=$(a_grid[a_i])",
                #      xlabel="Future Assets (a')", ylabel="Objective Value", lw=2, legend=false)


                # Define and optimize the objective function
                results[rho_i, a_i] = optimize(a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hhpar.beta * itp_cont_wrap(rho_grid[rho_i], a_prime)), 
                                            gpar.a_min, max_a_prime[rho_i, a_i], 
                                            GoldenSection()) 

                # Store results: Value and policy function
                V_new[rho_i, a_i] = -Optim.minimum(results[rho_i, a_i])
                policy_a[rho_i, a_i] = Optim.minimizer(results[rho_i, a_i]) 
            end
        end

        # --- Step 3: Check for Convergence ---
        max_error = maximum(abs.(V_new .- V_guess))
        # println("Iteration $iter, error: $max_error")

        if max_error < comp_params.vfi_tol
            # println("VFI converged after $iter iterations")
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
        policy_a_int   : Function or interpolation mapping (ρ, a) → a' (optimal asset policy).
        a_grid         : Vector of asset grid values.
        rho_grid       : Vector of productivity grid values.

    Returns:
        A matrix of policy values for l(ρ, a) or c(ρ, a), depending on the input.
    """

    # Preallocate policy matrix
    policy_matrix = zeros(length(rho_grid), length(a_grid))

    # Compute policy values at optimal a'
    for rho_i in 1:length(rho_grid)
        for a_i in 1:length(a_grid)
            a_prime_opt = policy_a_int(rho_grid[rho_i], a_grid[a_i])  # Get optimal a'
            policy_matrix[rho_i, a_i] = opt_policy_itp[rho_i, a_i](a_prime_opt)  # Get policy value
        end
    end

    return policy_matrix
end



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 5. SOLVING HOUSEHOLD PROBLEM #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function SolveHouseholdProblem(a_grid, rho_grid, l_grid, gpar, w, r, taxes, hhpar, pi_rho, comp_params)
    """
    Solves the household problem via value function iteration using first-order conditions (FOC).

    Returns:
        - hh_labor_taxes     : Net labor taxes y - T_y
        - hh_consumption     : Optimal consumption at each (ρ, a, a')
        - hh_consumption_tax : Consumption tax paid
        - opt_c_FOC          : Optimal consumption from FOC
        - opt_l_FOC          : Optimal labor from FOC
        - valuef             : Value function at convergence
        - policy_a           : Policy function for a'
        - policy_l           : Labor policy function l(ρ, a)
        - policy_c           : Consumption policy function c(ρ, a)
    """
    ########## PRELIMINARY ##########
    # Allocate net interest rate (for simplicity)
    net_r = (1 - taxes.tau_k)r

    ########## SECTION 1 - COMPUTE TAXES, CONSUMPTION AND UTILITY ##########

    # println("Solving budget constraint...")

    hh_labor_taxes, hh_consumption, hh_consumption_tax = compute_consumption_grid_for_itp(
        a_grid, rho_grid, l_grid, gpar, w, net_r, taxes;
        replace_neg_consumption = true
    )

    # cExp2cInt = interp_consumption(hh_consumption, hh_consumption_plus_tax)

    # test_budget_constraint() # TBM - to be removed or adjusted for function wrapping 

    ########## SECTION 2 - COMPUTE OPTIMAL CONSUMPTION AND LABOR ##########

    # println("Pinning down optimal labor and consumption using labor FOC...")

    opt_c_FOC, opt_l_FOC = find_opt_cons_labor(
        rho_grid, a_grid, w, net_r, taxes, hhpar, gpar;
        enforce_labor_cap = true,
        replace_neg_consumption = true
    )

    ########## SECTION 3 - INTERPOLATE POLICIES FROM FIRST ORDER CONDITIONS ##########

    opt_c_itp, opt_l_itp, opt_u_itp, max_a_prime = interp_opt_funs(
        a_grid, opt_c_FOC, opt_l_FOC, gpar, hhpar
    )

    ########## SECTION 4 - SOLVE VALUE FUNCTION ITERATION ##########

    # println("Launching VFI...")

    valuef, policy_a = intVFI_FOC_parallel(
        opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hhpar, gpar, comp_params
    )

    ########## SECTION 5 - RECONSTRUCT FINAL POLICY FUNCTIONS ##########

    policy_a_int = Spline2D_adj(rho_grid, a_grid, policy_a)

    policy_l = compute_policy_matrix(opt_l_itp, policy_a_int, a_grid, rho_grid)
    policy_c = compute_policy_matrix(opt_c_itp, policy_a_int, a_grid, rho_grid)

    ########## SECTION 6 - RUN CHECKS ON OUTPUT ##########
    # Check that value and policy functions contain only finite values
    for m in [valuef, policy_a, policy_c, policy_l]
        all(isfinite.(m)) ? nothing : error("Non-finite values in value or policy functions! Check VFI!")
    end
    
    return hh_labor_taxes, hh_consumption, hh_consumption_tax, opt_c_FOC, opt_l_FOC, valuef, policy_a, policy_l, policy_c
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------------# 6. COMPUTING AGGREGATES #-------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function stationary_distribution(a_grid, pi_rho, policy_a, gpar; tol=1e-9, max_iter=10000)    
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
            # println("Stationary Distribution: Converged after $iter iterations")
            return dist_new
        end
        dist .= dist_new
    end

    error("Stationary distribution did not converge")

    return dist_new
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 7. SOLVING GENERAL EQUILIBRIUM #---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


function capital_market_error(
    r, a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params
)
    # Wage implied by firm's FOC
    w = (1 - fpar.alpha) * fpar.tfp *
        ((fpar.alpha * fpar.tfp / (r + fpar.delta)) ^ (fpar.alpha / (1 - fpar.alpha)))

    # Household block
    (_, _, _, _, _, _, policy_a, policy_l, policy_c) =
        SolveHouseholdProblem(a_grid, rho_grid, l_grid, gpar, w, r, taxes, hhpar, pi_rho, comp_params)

    # Stationary distribution
    stat_dist = stationary_distribution(a_grid, pi_rho, policy_a, gpar; tol=1e-10, max_iter=10_000)

    # Aggregates
    asset_supply = sum(stat_dist * a_grid)
    labor_supply = sum(stat_dist .* policy_l)

    # Capital demand from firm's FOC
    asset_demand = ((fpar.alpha * fpar.tfp) / (r + fpar.delta)) ^ (1 / (1 - fpar.alpha)) * labor_supply

    return asset_demand - asset_supply
end


function ComputeEquilibrium_Roots(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params
)
    r_low = -fpar.delta
    r_high = 1 / hhpar.beta - 1

    f_root(r) = capital_market_error(r, a_grid, rho_grid, l_grid,
                                     gpar, hhpar, fpar, taxes,
                                     pi_rho, comp_params)

    r_eq = find_zero(f_root, (r_low, r_high), Brent(), atol=comp_params.ms_tol)

    # Recover equilibrium wage
    w_eq = (1 - fpar.alpha) * fpar.tfp *
           ((fpar.alpha * fpar.tfp / (r_eq + fpar.delta)) ^ (fpar.alpha / (1 - fpar.alpha)))

    println("✅ GE equilibrium found: r = $r_eq, w = $w_eq")

    return r_eq, w_eq
end

@elapsed ComputeEquilibrium_Roots(
    a_grid, rho_grid, l_grid,
    gpar, hhpar, fpar, taxes,
    pi_rho, comp_params
)