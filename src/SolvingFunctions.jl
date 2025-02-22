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


include("model_parameters.jl")
include("numerics.jl")
include("households.jl")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. TAXES AND UTILITY #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


compute_hh_taxes_consumption_utility = function(a_grid, N_a, rho_grid, l_grid, w, r, taxes, hh_parameters)

    #Compute gross labor income for each combination of labor and productivity
    y = (l_grid * rho_grid') .* w 

    # Compute taxation for each degree of progressivity - TBM: can be wrapped in a function
    # Expanding dimensions for broadcasting
    # Dims: X: labor, Y: productivity, Z: progressivity degree of labor income tax
    reshaped_y = ExpandMatrix(y, taxes.N_tau_y)
    Tau_y = Vector2NDimMatrix(taxes.tau_y, ndims(y))

    # Compute labor income taxes
    T_y = reshaped_y .* ones(1, 1, taxes.N_tau_y) .- taxes.lambda_y .* reshaped_y .^ (1 .- Tau_y);

    # Compute net labor income
    net_y = reshaped_y .- T_y

    # Compute disposable income after asset transfers (savings a' and interests (1+r)a)
    # Disposable income for each possible asset-state-specific interests yielded from t-1 
    # 4th dim: a
    y_after_interests = ExpandMatrix(net_y, N_a)
    interests = Vector2NDimMatrix((1 + r) .* a_grid, ndims(net_y))

    y_after_interests = y_after_interests .+ interests;

    # Find resource allocated to consumption (consumption + consumption taxes) for each combination of 
    # labor, productivity, degree of labor income progressivity, assets today
    # 5th dim: a_prime 

    consumption_expenditure = ExpandMatrix(y_after_interests, N_a)
    savings = Vector2NDimMatrix(a_grid, ndims(y_after_interests))

    consumption_expenditure = consumption_expenditure .- savings;

    # Disentangle consumption from consumption + consumption taxes (Feldstein specification)
    # for each degree of consumption tax progressivity
    # 6th dim: Degree of consumption tax progressivity (tau_c)

    hh_consumption_plus_tax = ExpandMatrix(consumption_expenditure, taxes.N_tau_c);

    # Initialise consumption matrix
    hh_consumption = copy(hh_consumption_plus_tax);

    # Find consumption level
    # ALLOWING FOR CONSUMPTION SUBSIDIES THROUGH CONSUMPTION TAX 
    # Comment the "; notax_upper = break_even" line to allow for redistributive subsidies
    # Through consumption tax

    @threads for i in 1:taxes.N_tau_c
        # Set progressivity rate
        prog_rate = taxes.tau_c[i]
        # Find break-even point 
        break_even = taxes.lambda_c^(1/prog_rate)
        # Find consumption
        # Assuming functional form with Tax-exemption area
        # To allow for redistributive subsidies remove the notax_upper argument from the function
        hh_consumption[:, :, :, :, :, i] .= find_c_feldstein.(hh_consumption_plus_tax[:, :, :, :, :, i], taxes.lambda_c, prog_rate
        ; notax_upper = break_even
        )
                                            
    end

    # Retrieve consumption tax
    hh_consumption_tax = hh_consumption_plus_tax .- hh_consumption;

    # Correct negative consumption 
    hh_consumption[hh_consumption .< 0] .= -Inf

    # Compute households utility
    hh_utility = copy(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    @threads for l in 1:N_l
        hh_utility[l, :, :, :, :, :] .= ifelse.(hh_consumption[l, :, :, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :, :, :],
                                                l_grid[l], hh_parameters.rra, hh_parameters.phi, hh_parameters.frisch), 
                                                hh_utility[l, :, :, :, :, :])
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_utility
end



### Simpler version: single rate of progressivity for consumption taxes and labor income taxes

compute_hh_taxes_consumption_utility_ = function(a_grid, N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters)

    #Compute gross labor income for each combination of labor and productivity
    y = (l_grid * rho_grid') .* w 

    # Compute labor income taxes for one degree of labor income tax progressivity - Allowing negative tax
    T_y = y .- taxes.lambda_y .* y .^ (1 - Tau_y);

    # Correct for negative tax
    # T_y = max.(T_y, 0)

    # Compute net labor income
    net_y = y .- T_y

    # Compute disposable income after asset transfers (savings a' and interests (1+r)a)
    # Disposable income for each possible asset-state-specific interests yielded from t-1 
    # 3rd dim: a
    y_after_interests = ExpandMatrix(net_y, N_a)
    interests = Vector2NDimMatrix((1 + r) .* a_grid, ndims(net_y))

    y_after_interests = y_after_interests .+ interests;

    # Find resource allocated to consumption (consumption + consumption taxes) for each combination of 
    # labor, productivity, assets today
    # 4th dim: a_prime 

    consumption_expenditure = ExpandMatrix(y_after_interests, N_a)
    savings = Vector2NDimMatrix(a_grid, ndims(y_after_interests))

    consumption_expenditure = consumption_expenditure .- savings;

    # Disentangle consumption from consumption + consumption taxes (Feldstein specification)
    # for one degree of consumption tax progressivity

    hh_consumption_plus_tax = copy(consumption_expenditure)

    # Initialise consumption matrix
    hh_consumption = copy(hh_consumption_plus_tax);

    # Find consumption level
    # ALLOWING FOR CONSUMPTION SUBSIDIES THROUGH CONSUMPTION TAX 
    # Comment the "; notax_upper = break_even" line to allow for redistributive subsidies
    # Through consumption tax

    # Set progressivity rate
    prog_rate = Tau_c

    # Find break-even point 
    break_even = taxes.lambda_c^(1/prog_rate)

    # To allow for redistributive subsidies remove the notax_upper argument from the function
    hh_consumption[:, :, :, :] .= find_c_feldstein.(hh_consumption_plus_tax[:, :, :, :], taxes.lambda_c, prog_rate
    ; notax_upper = break_even
    )

    # Retrieve consumption tax
    hh_consumption_tax = hh_consumption_plus_tax .- hh_consumption;

    # Correct negative consumption 
    hh_consumption[hh_consumption .< 0] .= -Inf

    # Compute households utility
    hh_utility = copy(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    @threads for l in 1:N_l
        hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :],
                                                l_grid[l], hh_parameters.rra, hh_parameters.phi, hh_parameters.frisch), 
                                                hh_utility[l, :, :, :])
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_utility
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# 2. VFI #----------------------------------#
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

function standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
    # Initialize the state-dependent value function guess (over rho and a)
    V_guess = zeros(N_rho, N_a)
    
    # Preallocate the policy arrays for asset and labor (both 2-D: (rho, a))
    policy_a_index = zeros(Int64, N_rho, N_a)
    policy_l_index = zeros(Int64, N_rho, N_a)
    
    # Preallocate candidate arrays for each labor option.
    candidate = zeros(N_rho, N_a, N_a)
    # V_candidate will hold the value (after optimizing over a') for each labor option.
    V_candidate = zeros(N_l, N_rho, N_a)
    asset_policy_candidate = zeros(Int64, N_l, N_rho, N_a)
    
    # Preallocate a new state-value function for the iteration.
    V_new = similar(V_guess)

    # Continuation value
    cont = zeros(N_rho, N_a)
    
    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Compute the continuation value for every state ---
        # cont[rho, j] = sum_{rho'} pi_rho[rho, rho'] * V_guess[rho', j]
        cont .= pi_rho * V_guess  # shape: (N_rho, N_a)
        
        # --- Step 2: For each labor option, compute candidate value functions ---
        # The candidate for labor option l is:
        #   candidate(l, rho, a, j) = hh_utility(l, rho, a, j) + beta * cont[rho, j]
        # We need to maximize over j (next asset choices) for each (rho, a).
        # Loop over labor options; the remaining maximization will be vectorized over (rho, a).
        for l in 1:N_l
            # hh_utility[l, :, :, :] has shape (N_rho, N_a, N_a)
            # Reshape cont to (N_rho, 1, N_a) so it broadcasts along the a dimension.
            candidate .= hh_utility[l, :, :, :] .+ hh_parameters.beta .* reshape(cont, (N_rho, 1, N_a))
            # For each current state (rho, a), maximize over the last dimension (j).
            # We'll loop over (rho, a) to extract both max values and argmax indices.
            for rho in 1:N_rho
                for a in 1:N_a
                    # findmax returns (max_value, index) for candidate[l, rho, a, :]
                    value, idx = findmax(candidate[rho, a, :])
                    V_candidate[l, rho, a] = value
                    asset_policy_candidate[l, rho, a] = idx
                end
            end
        end
        
        # --- Step 3: Collapse the labor dimension ---
        # For each state (rho, a), choose the labor option that gives the highest candidate value.
        for rho in 1:N_rho
            for a in 1:N_a
                value, labor_opt = findmax(V_candidate[:, rho, a])
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

function MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
    # Initialize the state-dependent value function guess (over ρ and a)
    V_guess = zeros(N_rho, N_a)
    
    # Preallocate the policy arrays (for state (ρ, a))
    policy_a_index = zeros(Int64, N_rho, N_a)
    policy_l_index = zeros(Int64, N_rho, N_a)
    
    # Preallocate candidate arrays (per labor option)
    candidate = zeros(N_rho, N_a, N_a)
    # V_candidate[l, ρ, a] = optimal value for labor option l at state (ρ,a)
    V_candidate = zeros(N_l, N_rho, N_a)
    # asset_policy_candidate[l, ρ, a] stores the argmax over next-assets for that labor option.
    asset_policy_candidate = zeros(Int64, N_l, N_rho, N_a)

    # Continuation value
    cont = zeros(N_rho, N_a)
    max_vals    = zeros(N_rho, N_a)
    argmax_vals = zeros(N_rho, N_a)
    
    # Preallocate the new state value function
    V_new = similar(V_guess)
    
    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Compute continuation value ---
        # cont[ρ, j] = Σ_{ρ'} π(ρ,ρ') V_guess(ρ', j)
        cont .= pi_rho * V_guess  # (N_rho, N_a)
        
        # --- Step 2: For each labor option, compute candidate value functions ---
        # For each labor option l, hh_utility[l, :, :, :] has shape (N_rho, N_a, N_a),
        # and we add beta*cont (reshaped to (N_rho,1,N_a)) along the asset-choice dimension.
        for l in 1:N_l
            candidate .= hh_utility[l, :, :, :] .+ hh_parameters.beta .* reshape(cont, (N_rho, 1, N_a))
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
        policy_a_index .= [asset_policy_candidate[policy_l_index[i,j], i, j] for i in 1:N_rho, j in 1:N_a]

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
