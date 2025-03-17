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

function compute_hh_taxes_consumption_utility_ME(a_grid, rho_grid, l_grid, gpar, w, r, taxes, hh_parameters)

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
    gross_capital_returns = Vector2NDimMatrix((1 + (1 - taxes.tau_k)r) .* a_grid, old_dims_y)

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
                                                l_grid[l], hh_parameters), 
                                                hh_consumption[l, :, :, :])
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_utility
end

###############################################################################
##### SPLIT AND WRITE TO DISK TO SAVE MEMORY #####
###############################################################################


function compute_consumption_grid(a_grid, rho_grid, l_grid, gpar, w, r, taxes)
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
    gross_capital_returns = Vector2NDimMatrix((1 + (1 - taxes.tau_k)r) .* a_grid, old_dims_y)

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


function compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, taxes; replace_neg_consumption = false)
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
    gross_capital_returns = Vector2NDimMatrix((1 + (1 - taxes.tau_k)r) .* a_grid, old_dims_y)

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

    # Retrieve consumption tax - in-place to avoid costly memory allocation
    hh_consumption_tax = hh_consumption_plus_tax .- hh_consumption;

    # Correct negative consumption 
    if replace_neg_consumption == true
        @views hh_consumption[hh_consumption .< 0] .= -Inf
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_consumption_plus_tax
end

function compute_utility_grid(hh_consumption, l_grid, hh_parameters; minus_inf = true)
        ########## SECTION 3 - COMPUTE HOUSEHOLD UTILITY ##########

    # Compute households utility
    hh_utility = similar(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    @threads for l in 1:gpar.N_l        
        @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :],
                                                l_grid[l], hh_parameters), 
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

function find_opt_cons_labor(rho_grid, a_grid, w, net_r, taxes, hh_parameters, gpar)
    # Create matrices for optimal consumption and optimal labor
    opt_consumption = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)
    opt_labor = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)

    #Pre-allocate vars
    rho = 0
    rhs = 0 
    opt_c = 0

    for rho_i in 1:gpar.N_rho
        # Get rho from the grid
        rho = rho_grid[rho_i]
    
        # Define the wage as a function of c using the labor supply function
        wage_star(c) = rho * w * get_opt_labor_from_FOC(c, rho, w, taxes, hh_parameters)
    
        for a_i in 1:gpar.N_a
            for a_prime_i in 1:gpar.N_a
                # Compute saving returns + saving expenditure (rhs)
                rhs = (1 + net_r) * a_grid[a_i] - a_grid[a_prime_i]
    
                # Define the objective function to solve for optimal consumption (c)
                f = c -> 2 * c - taxes.lambda_c * c^(1 - taxes.tau_c) - taxes.lambda_y * wage_star(c) ^ (1 - taxes.tau_y) - rhs
    
                try
                    # Find solution, if any
                    opt_c = find_zero(f, 0.5)
                    @views opt_consumption[rho_i, a_i, a_prime_i] = opt_c #0.5 Initial guess, adjustable
                    # Get optimal labor
                    @views opt_labor[rho_i, a_i, a_prime_i] = get_opt_labor_from_FOC(opt_c, rho, w, taxes, hh_parameters)
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
    return opt_consumption, opt_labor
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# 3. VFI #----------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Standard baseline function - Vectorised, no parallelisation
# Attempts to parallelize did not improve performances
# Benchmarking results, state space with N_l = 10, N_rho = 7, N_a = 100, N_tau_c=N_tau_y=1
# Speed 2.90 s
# Memory 3.01 GB
# Benchmarking results, state space with N_l = 50, N_rho = 7, N_a = 300, N_tau_c=N_tau_y=1
# Speed 176.096 s
# Memory 126.66 GB

function standardVFI(gpar, comp_params, hh_parameters, hh_utility, pi_rho)
    # Initialize the state-dependent value function guess (over rho and a)
    V_guess = zeros(gpar.N_rho, gpar.N_a)
    
    # Preallocate the policy arrays for asset and labor (both 2-D: (rho, a))
    policy_a_index = zeros(Int64, gpar.N_rho, gpar.N_a)
    policy_l_index = zeros(Int64, gpar.N_rho, gpar.N_a)
    
    # Preallocate candidate arrays for each labor option.
    candidate = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)
    # V_candidate will hold the value (after optimizing over a') for each labor option.
    V_candidate = zeros(gpar.N_l, gpar.N_rho, gpar.N_a)
    asset_policy_candidate = zeros(Int64, gpar.N_l, gpar.N_rho, gpar.N_a)
    
    # Preallocate a new state-value function for the iteration.
    V_new = similar(V_guess)

    # Continuation value
    cont = zeros(gpar.N_rho, gpar.N_a)
    
    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Compute the continuation value for every state ---
        # cont[rho, j] = sum_{rho'} pi_rho[rho, rho'] * V_guess[rho', j]
        cont .= pi_rho * V_guess  # shape: (N_rho, N_a)
        
        # --- Step 2: For each labor option, compute candidate value functions ---
        # The candidate for labor option l is:
        #   candidate(l, rho, a, j) = hh_utility(l, rho, a, j) + beta * cont[rho, j]
        # We need to maximize over j (next asset choices) for each (rho, a).
        # Loop over labor options; the remaining maximization will be vectorized over (rho, a).
        for l in 1:gpar.N_l            # hh_utility[l, :, :, :] has shape (N_rho, N_a, N_a)
            # Reshape cont to (N_rho, 1, N_a) so it broadcasts along the a dimension.
            candidate .= hh_utility[l, :, :, :] .+ hh_parameters.beta .* reshape(cont, (gpar.N_rho, 1, gpar.N_a))
            # For each current state (rho, a), maximize over the last dimension (j).
            # We'll loop over (rho, a) to extract both max values and argmax indices.
            for rho in 1:gpar.N_rho                for a in 1:gpar.N_a                    # findmax returns (max_value, index) for candidate[l, rho, a, :]
                    value, idx = findmax(candidate[rho, a, :])
                    V_candidate[l, rho, a] = value
                    asset_policy_candidate[l, rho, a] = idx
                end
            end
        end
        
        # --- Step 3: Collapse the labor dimension ---
        # For each state (rho, a), choose the labor option that gives the highest candidate value.
        for rho in 1:gpar.N_rho            for a in 1:gpar.N_a                value, labor_opt = findmax(V_candidate[:, rho, a])
                V_new[rho, a] = value
                policy_l_index[rho, a] = labor_opt
                policy_a_index[rho, a] = asset_policy_candidate[labor_opt, rho, a]
            end
        end
        
        # --- Step 4: Check convergence ---
        if maximum(abs.(V_new .- V_guess)) < comp_params.vfi_tol
            println("Converged after $iter iterations")
            break
        end
        
        # Update the guess for the next iteration.
        V_guess .= V_new
    end
    
    return V_new, policy_a_index, policy_l_index
end


#### Memory efficient version!!! Using more vectorisation and pre-allocation to save memory 
# Benchmarking results, state space with N_l = 10, N_rho = 7, N_a = 100, N_tau_c=N_tau_y=1
# Speed 3.02 s
# Memory 1.48 GB
# Benchmarking results, state space with N_l = 50, N_rho = 7, N_a = 300, N_tau_c=N_tau_y=1
# Speed 158.803 s
# Memory 63.25 GB

function MemoryEffVFI(gpar, comp_params, hh_parameters, hh_utility, pi_rho; V_guess_read = nothing)
    if isnothing(V_guess_read) 
        # Initialize the state-dependent value function guess (over ρ and a)
        V_guess = zeros(gpar.N_rho, gpar.N_a)
    else
        V_guess = V_guess_read 
    end
    
    # Preallocate the policy arrays (for state (ρ, a))
    policy_a_index = zeros(Int64, gpar.N_rho, gpar.N_a)
    policy_l_index = zeros(Int64, gpar.N_rho, gpar.N_a)
    
    # Preallocate candidate arrays (per labor option)
    candidate = zeros(gpar.N_rho, gpar.N_a, gpar.N_a)
    # V_candidate[l, ρ, a] = optimal value for labor option l at state (ρ,a)
    V_candidate = zeros(gpar.N_l, gpar.N_rho, gpar.N_a)
    # asset_policy_candidate[l, ρ, a] stores the argmax over next-assets for that labor option.
    asset_policy_candidate = zeros(Int64, gpar.N_l, gpar.N_rho, gpar.N_a)

    # Continuation value
    cont = zeros(gpar.N_rho, 1, gpar.N_a)
    max_vals    = zeros(gpar.N_rho, gpar.N_a)
    argmax_vals = zeros(gpar.N_rho, gpar.N_a)
    
    # Preallocate the new state value function
    V_new = similar(V_guess)
    
    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Compute continuation value ---
        # cont[ρ, j] = Σ_{ρ'} π(ρ,ρ') V_guess(ρ', j)
        cont .= reshape(pi_rho * V_guess, (gpar.N_rho, 1, gpar.N_a))  # (N_rho, N_a) but reshaped for computation
        
        # --- Step 2: For each labor option, compute candidate value functions ---
        # For each labor option l, hh_utility[l, :, :, :] has shape (N_rho, N_a, N_a),
        # and we add beta*cont (reshaped to (N_rho,1,N_a)) along the asset-choice dimension.
        @inbounds for l in 1:gpar.N_l            @views candidate .= hh_utility[l, :, :, :] .+ hh_parameters.beta .* cont
            # candidate now has shape (N_rho, N_a, N_a), where the 3rd dimension indexes next assets.
            # Vectorize the maximization over next assets:
            V_candidate[l, :, :] .= dropdims(maximum(candidate, dims=3), dims=3)
            asset_policy_candidate[l, :, :] .= map(x -> x[3], dropdims(argmax(candidate, dims=3), dims=3))            
        end
        
        # --- Step 3: Collapse the labor dimension ---
        # Vectorize over (ρ, a) using maximum/argmax:
        max_vals    .= dropdims(maximum(V_candidate, dims=1), dims=1)  # (N_rho, N_a)
        argmax_vals = dropdims(argmax(V_candidate, dims=1), dims=1)   # (N_rho, N_a)

        V_new .= max_vals
        policy_l_index .= map(x -> x[1], argmax_vals)


        # Extract the asset policy corresponding to the optimal labor choice.
        # Option 1: Pure vectorized comprehension:
        policy_a_index .= [asset_policy_candidate[policy_l_index[i,j], i, j] for i in 1:gpar.N_rho, j in 1:gpar.N_a]

        # Option 2: Parallelized version using Threads:
        # Uncomment the following block if you want to parallelize:
        # policy_a_index = similar(policy_l_index)
        # Threads.@threads for idx in eachindex(policy_l_index)
        #     i, j = ind2sub(size(policy_l_index), idx)
        #     policy_a_index[idx] = asset_policy_candidate[argmax_vals[i,j], i, j]
        # end

        # --- Step 4: Check convergence ---
        if maximum(abs.(V_new .- V_guess)) < comp_params.vfi_tol
            println("Converged after $iter iterations")
            break
        end
        
        # Update the guess.
        V_guess .= V_new
    end
    
    return V_new, policy_a_index, policy_l_index
end

############################## INTERPOLATED VFI ###############################

function intVFI(hh_consumption, l_grid, rho_grid, a_grid, hh_parameters, comp_params, 
    pi_rho, g_par)
    # Initialize the state-dependent value function guess (over ρ and a)
    V_guess = zeros(gpar.N_rho, gpar.N_a)

    # Preallocate the policy arrays (for state (ρ, a))
    policy_a = zeros(Float64, gpar.N_rho, gpar.N_a)
    policy_l = zeros(Float64, gpar.N_rho, gpar.N_a)
    
    # # V_candidate[l, ρ, a] = optimal value for labor option l at state (ρ,a)
    Vcand = zeros(gpar.N_l, gpar.N_rho, gpar.N_a)
    policy_a_opt = zeros(gpar.N_l, gpar.N_rho, gpar.N_a)

    # # Continuation value
    cont = zeros(gpar.N_rho, gpar.N_a)
    
    # Preallocate the new state value function
    V_new = similar(V_guess)
    
    # Create interpolant for household utility
    hh_utility = compute_utility_grid(hh_consumption, l_grid, hh_parameters; minus_inf = true)
    # Replace -Inf with large negative values to ensure smoothness of interpolation
    # itp_utility = extrapolate(interpolate((l_grid, rho_grid, a_grid, a_grid), hh_utility, Gridded(Linear())), Interpolations.Flat())
    # utility_interp = (l, rho, a, a_prime) -> itp_utility(l, rho, a, a_prime)

    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Interpolate continuation value ---
        itp_cont, itp_cont_wrap = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)

        # --- Step 2: For each labor option, compute candidate value functions ---
        # For each labor option l, maximise the interpolated objective function
        @inbounds for l in 1:gpar.N_l            
            for rho in 1:gpar.N_rho                
                for a in 1:gpar.N_a                    
                    # Interpolate utility for given l, rho and a
                    # utility_interp = extrapolate(interpolate((a_grid,), hh_utility[l, rho, a, :], Gridded(Linear())), Interpolations.Flat())
                    # try
                    utility_interp, max_a_prime = piecewise_1D_interpolation(a_grid, hh_utility[l, rho, a, :]; 
                                                                             spline = false, return_threshold = true)
                    # try
                    # utility_interp, max_a_prime = piecewise_1D_interpolation(a_grid, hh_utility[l, rho, a, :]; 
                    #                                                             spline=false, return_threshold=true)
                    # catch e
                    #     println("Error encountered at indices: l = ", l, ", rho = ", rho, ", a = ", a)
                    #     println("Error message: ", e)
                    #     rethrow()  # Optional: Rethrow the error to stop execution, or remove if you want it to continue
                    # end

                    # plot_interpolation(a_grid, hh_utility[l, rho, a, :], utility_interp, x_max=2.5)
                    # Define objective function to maximise
                    objective = a_prime -> -(utility_interp(a_prime) + hh_parameters.beta * itp_cont_wrap(rho, a_prime))

                    # Optimize - Restrict search to feasible points to ensure finding right solution
                    result = optimize(objective, gpar.a_min, max_a_prime, Brent()) #TBM - Check Brent()
                    
                    # Temporary check - Ensure no infinite value is stored
                    # if isinf(Optim.minimum(result))
                    #     error("Error: Solution is Inf, check process!")
                    # end

                    # Store the candidate value for this labor option.
                    @views Vcand[l, rho, a] =  -Optim.minimum(result)
                    
                    # Also store the optimal a' for this labor option.
                    @views policy_a_opt[l, rho, a] = Optim.minimizer(result) 
                        
                    # catch e
                    #     println("Error encountered at indices: l = ", l, ", rho = ", rho, ", a = ", a)
                    #     println("Error message: ", e)
                    # end
                end
            end
        end
        
        # --- Step 3: Interpolate Over Labor (for each (ρ, a)) ---
        # Vcand_interp = Dict()        # Store splines per V(ρ, a)
        # policy_a_interp = Dict()     # Store splines for policy_a_opt
        # Vcand_interp = Dict{Tuple{Int, Int}, Spline1D}()
        # policy_a_interp = Dict{Tuple{Int, Int}, Spline1D}()


        # # Construct cubic splines over l for each (ρ, a) pair
        # @inbounds for rho in 1:gpar.N_rho            
        #     for a in 1:gpar.N_a                
        #         Vcand_interp[(rho, a)] = Spline1D(l_grid, Vcand[:, rho, a], k=3)
        #         policy_a_interp[(rho, a)] = Spline1D(l_grid, policy_a_opt[:, rho, a], k=3)
        #     end
        # end

        # --- Step 3: Interpolate Over Labor (for each (ρ, a)) ---
        Vcand_interp = Dict{Tuple{Int, Int}, Any}()  # Store linear interpolators per V(ρ, a)
        policy_a_interp = Dict{Tuple{Int, Int}, Any}()  # Store linear interpolators for policy_a_opt

        # Construct linear interpolations over l for each (ρ, a) pair
        @inbounds for rho in 1:gpar.N_rho            
            for a in 1:gpar.N_a
                # Linear interpolation for Vcand
                Vcand_interp[(rho, a)] = extrapolate(interpolate((l_grid,), Vcand[:, rho, a], Gridded(Linear())), Interpolations.Flat())

                # Linear interpolation for policy_a_opt
                policy_a_interp[(rho, a)] = extrapolate(interpolate((l_grid,), policy_a_opt[:, rho, a], Gridded(Linear())), Interpolations.Flat())
            end
        end
        
        # # Retrieve the interpolator for policy function
        # rho_val, a_val = 3, 20
        # policy_interp = policy_a_interp[(rho_val, a_val)]

        # # Generate a finer grid for labor values
        # fine_l_grid = range(minimum(l_grid), maximum(l_grid), length=200)  # More points for smooth curve
        # interp_values = policy_interp.(fine_l_grid)  # Evaluate interpolation

        # # Plot original data points
        # scatter(l_grid, policy_a_opt[:, rho_val, a_val], markersize=4, color=:red, label="Original Data")

        # # Plot interpolated spline
        # plot!(fine_l_grid, interp_values, linewidth=2, color=:blue, label="Cubic Spline Interpolation", legend=false)

        # # Titles and labels
        # title!("Interpolated Policy Function (ρ=$rho_val, a=$a_val)")
        # xlabel!("Labor Grid (l)")
        # ylabel!("Policy a'")

        # --- Step 4: Solve for Optimal Labor Choice ---
        @inbounds for rho in 1:gpar.N_rho            
            for a in 1:gpar.N_a                
                # Define the objective function to maximize over labor
                obj_l = l -> -Vcand_interp[(rho, a)](l)  # Interpolate Vcand over l

                # Optimize labor choice in [l_min, l_max]
                res_l = optimize(obj_l, gpar.l_min, gpar.l_max, Brent())

                # Store optimal labor choice (continuous)
                policy_l[rho, a] = Optim.minimizer(res_l)

                # Compute corresponding optimal asset choice
                l_star = policy_l[rho, a]
                policy_a[rho, a] = policy_a_interp[(rho, a)](l_star)

                # Store updated value function
                V_new[rho, a] = -Optim.minimum(res_l)
            end
        end

        # --- Step 4: Check convergence ---
        max_error = maximum(abs.(V_new .- V_guess))
        if max_error < comp_params.vfi_tol
            println("Converged after $iter iterations")
            break
        end

        # Otherwise, update the guess.
        println("Iteration $iter, error: $max_error")
        V_guess .= V_new
        
    end
    
    return V_new, policy_a, policy_l
end


######################### VFI - EXPLOITING LABOR FOC ##########################


function intVFI_FOC(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hh_parameters, gpar, comp_params)
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
                objective = a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hh_parameters.beta * itp_cont_wrap(rho_grid[rho_i], a_prime))
    
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


function intVFI_FOC_parallel(opt_u_itp, pi_rho, rho_grid, a_grid, max_a_prime, hh_parameters, gpar, comp_params)
    """
    Performs Value Function Iteration (VFI) using optimal labor and consumption choices 
    derived from the first-order conditions.

    Args:
        opt_u_itp     : Interpolated utility function {(ρ, a) => u(c(a'), l(a'))}
        pi_rho        : Transition matrix for productivity levels
        rho_grid      : Grid of productivity levels
        a_grid        : Grid of asset values
        max_a_prime   : Upper bound for a' choices per (ρ, a)
        hh_parameters : Household parameters (contains β)
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
            # Define and optimize the objective function
            results[rho_i, a_i] = optimize(a_prime -> -(opt_u_itp[rho_i, a_i](a_prime) + hh_parameters.beta * itp_cont_wrap(rho_grid[rho_i], a_prime)), 
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
            println("Converged after $iter iterations")
            break
        end

        # Update guess for next iteration
        V_guess .= V_new
    end
    
    return V_new, policy_a
end



            