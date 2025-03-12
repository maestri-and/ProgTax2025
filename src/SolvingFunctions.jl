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
#---------------# 0. TAXES AND UTILITY #---------------#
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

    # Compute disposable income after asset transfers (gross capital returns (1+r)a)
    # Disposable income for each possible asset-state-specific interests yielded from t-1 
    # 3rd dim: a
    old_dims_y = ndims(y)
    y = ExpandMatrix(y, gpar.N_a)
    gross_capital_returns = Vector2NDimMatrix((1 + r) .* a_grid, old_dims_y)

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
    ; notax_upper = break_even
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
                                                l_grid[l], hh_parameters.rra, hh_parameters.phi, hh_parameters.frisch), 
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
    gross_capital_returns = Vector2NDimMatrix((1 + r) .* a_grid, old_dims_y)

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
    ; notax_upper = break_even
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


function compute_consumption_grid_for_itp(a_grid, rho_grid, l_grid, gpar, w, r, taxes)
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
    gross_capital_returns = Vector2NDimMatrix((1 + r) .* a_grid, old_dims_y)

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
    notax_upper = break_even
    )

    # Retrieve consumption tax - in-place to avoid costly memory allocation
    hh_consumption_tax = hh_consumption_plus_tax .- hh_consumption;

    # Correct negative consumption 
    @views hh_consumption[hh_consumption .< 0] .= -Inf

    return T_y, hh_consumption, hh_consumption_tax, hh_consumption_plus_tax
end

function compute_utility_grid(hh_consumption, l_grid, hh_parameters; minus_inf = true)
        ########## SECTION 3 - COMPUTE HOUSEHOLD UTILITY ##########

    # Compute households utility
    hh_utility = similar(hh_consumption); # Pre-allocate

    # Compute household utility if consumption is positive
    @threads for l in 1:gpar.N_l        @views hh_utility[l, :, :, :] .= ifelse.(hh_consumption[l, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :],
                                                l_grid[l], hh_parameters.rra, hh_parameters.phi, hh_parameters.frisch), 
                                                hh_consumption[l, :, :, :])
    end

    if !minus_inf 
        hh_utility[hh_utility .== -Inf] .= minimum(hh_utility[hh_utility .!= -Inf]) - 1
    end
    return hh_utility
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
    hh_utility = compute_utility_grid(hh_consumption, l_grid, hh_parameters; minus_inf = false)
    # Replace -Inf with large negative values to ensure smoothness of interpolation
    # itp_utility = extrapolate(interpolate((l_grid, rho_grid, a_grid, a_grid), hh_utility, Gridded(Linear())), Interpolations.Flat())
    # utility_interp = (l, rho, a, a_prime) -> itp_utility(l, rho, a, a_prime)
      

    for iter in 1:comp_params.vfi_max_iter
        # --- Step 1: Interpolate continuation value ---
        itp_cont, cont_interp = interp_cont_value(V_guess, pi_rho, rho_grid, a_grid)

        # --- Step 2: For each labor option, compute candidate value functions ---
        # For each labor option l, maximise the interpolated objective function
        @inbounds for l in 1:gpar.N_l            
            for rho in 1:gpar.N_rho                
                for a in 1:gpar.N_a                    
                    # Interpolate utility for given l, rho and a
                    utility_interp = extrapolate(interpolate((a_grid,), hh_utility[l, rho, a, :], Gridded(Linear())), Line())
                    # utility_interp = Spline1D(a_prime_values, hh_utility[l, rho, a, :], k=3) # Dierckx - jump issues!
                    

                    # plot_interpolation(a_grid, hh_utility[l, rho, a, :], utility_interp, x_max=2.5)
                    # Define objective function to maximise
                    objective = a_prime -> -(utility_interp(a_prime) + hh_parameters.beta * cont_interp(a_prime, rho))

                    # Optimize
                    result = optimize(objective, gpar.a_min, gpar.a_max, Brent()) #TBM - Check Brent()
                        
                    # Store the candidate value for this labor option.
                    @views Vcand[l, rho, a] =  -Optim.minimum(result)
                    # Also store the optimal a' for this labor option.
                    @views policy_a_opt[l, rho, a] = Optim.minimizer(result) 
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
                Vcand_interp[(rho, a)] = extrapolate(interpolate((l_grid,), Vcand[:, rho, a], Gridded(Linear())), Line())

                # Linear interpolation for policy_a_opt
                policy_a_interp[(rho, a)] = extrapolate(interpolate((l_grid,), policy_a_opt[:, rho, a], Gridded(Linear())), Line())
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
        if maximum(abs.(V_new .- V_guess)) < comp_params.vfi_tol
            println("Converged after $iter iterations")
            break
        end
        
        # Otherwise, update the guess.
        V_guess .= V_new
    end
    
    return V_new, policy_a, policy_l
end
