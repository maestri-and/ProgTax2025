###############################################################################
############################### OLDFUNCTIONS.JL ###############################

####### This script temporarily collects old functions (to be removed) ########
############### the used to solve benchmark ProgTax(2025) model ###############

###############################################################################



###############################################################################
############################## FROM RUNMODEL.JL ###############################
###############################################################################


#### CUSTOM GRIDS ####
# Custom labor grid
# coarse1 = 1/25    # proportion of points in [0, 0.1]
# dense   = 23/25   # proportion of points in [0.1, 0.5]
# coarse2 = 1/25    # proportion of points in [0.5, 1.0]

# l_grid_low   = range(0.0, stop=0.1, length=Int(coarse1*N_l))
# l_grid_dense = range(0.1, stop=0.5, length=Int(dense*N_l+2))
# l_grid_high  = range(0.5, stop=1.0, length=Int(coarse2*N_l))

# l_grid = vcat(l_grid_low, l_grid_dense[2:end-1], l_grid_high)

####### COMPUTING FULL CONSUMPTION GRID ########

# Original version - including also tax progressivity rates
# @elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility(a_grid, 
#                                                                     N_a, rho_grid, l_grid, w, r, taxes, hh_parameters);

# Simplified version for one degree of progressivity of labor income and consumption taxes
# @elapsed hh_labor_taxes, hh_consumption, hh_consumption_tax, hh_utility = compute_hh_taxes_consumption_utility_ME(a_grid, 
#                                                                     N_a, rho_grid, l_grid, N_l, w, r, Tau_y, Tau_c, taxes, hh_parameters);

# @benchmark compute_hh_taxes_consumption_utility_ME(a_grid, N_a, rho_grid, l_grid, w, r, Tau_y, Tau_c, taxes, hh_parameters)

# Split operations to save memory 
# @elapsed T_y, hh_consumption = compute_consumption_grid(a_grid, rho_grid, l_grid, N_a, N_rho, N_l, w, r, Tau_y, Tau_c, taxes)
# # Sys.free_memory() |> Base.format_bytes
# # varinfo()
# GC.gc()
# @elapsed @views hh_consumption .= compute_utility_grid(hh_consumption, l_grid, hh_parameters)

# # Rename for clarity 
# hh_utility = hh_consumption

# # Benchmark 
# @benchmark begin
#     T_y, hh_consumption = compute_consumption_grid(a_grid, rho_grid, l_grid, N_a, N_rho, N_l, w, r, Tau_y, Tau_c, taxes)
#     @views hh_consumption .= compute_utility_grid(hh_consumption, l_grid, hh_parameters)
# end

# @benchmark compute_hh_taxes_consumption_utility_ME(a_grid, N_a, rho_grid, l_grid, N_l, w, r, Tau_y, Tau_c, taxes, hh_parameters)

#---------------------# PERFORM VFI - VECTORISED VERSION #--------------------#

# @elapsed V_new, policy_a_index, policy_l_index = standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)
#@benchmark standardVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)

# @elapsed V_new, policy_a_index, policy_l_index = MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho;
#                         # V_guess_read = ReadMatrix("output/preliminary/V_guess_matrix.txt")
#                         )
# @benchmark MemoryEffVFI(N_l, N_rho, N_a, comp_params, hh_parameters, hh_utility, pi_rho)



################## COMPUTING CONSUMPTION, TAXES AND UTILITY ###################

# This script computes consumption, consumption taxes and utility for each 
# possible combination of 
# 1. Labor
# 2. Productivity
# 3. Labor income tax progressivity degree
# 4. Assets today
# 5. Assets tomorrow
# 6. Consumption tax progressivity degree

function compute_hh_taxes_consumption_utility_full(a_grid, N_a, rho_grid, l_grid, w, r, taxes, hh_parameters)

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
    @threads for l in 1:gpar.N_l        hh_utility[l, :, :, :, :, :] .= ifelse.(hh_consumption[l, :, :, :, :, :] .> 0,
                                                get_utility_hh.(hh_consumption[l, :, :, :, :, :],
                                                l_grid[l], hh_parameters.rra, hh_parameters.phi, hh_parameters.frisch), 
                                                hh_utility[l, :, :, :, :, :])
    end

    return T_y, hh_consumption, hh_consumption_tax, hh_utility
end


########################### VALUE FUNCTION ITERATION ###########################


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




################ EX-POST INTERPOLATIONS ##################

println("Interpolating results...")
###### Interpolate Asset policy function ######

fine_grid_a = range(gpar.a_min, gpar.a_max, length=2*gpar.N_a) 

# Store interpolated values for plotting
interp_policy_a = zeros(size(policy_a, 1), length(fine_grid_a))
a_grid_r = range(gpar.a_min, gpar.a_max, gpar.N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = policy_a[rho_i, :]

    # Create the cubic spline interpolant
    itp = cubic_spline_interpolation(a_grid_r, policy_values, extrapolation_bc=Interpolations.Flat())

    # Evaluate the interpolant on the fine grid
    interp_policy_a[rho_i, :] = itp.(fine_grid_a)
end

# Plot
# Define colors for each productivity level
colors = palette(:viridis, size(policy_a, 1));

# Plot the interpolated policy functions
pfa_int = plot(
    fine_grid_a, interp_policy_a[1, :],
    label="ρ = $(rho_grid[1])", color=colors[1], linewidth=2,
    xlabel = "Assets (a)", ylabel = "Next Period Assets (a')",
    title = "Policy Functions for Assets",
    legend = :bottomright);

for rho_i in 2:gpar.N_rho    plot!(pfa_int, fine_grid_a, interp_policy_a[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfa_int)

# savefig(pfa_int, "output/preliminary/asset_policy_int_len$gpar.N_a.png")



###### Interpolate Labor policy function ######

# Store interpolated values for plotting
interp_policy_l = zeros(size(policy_l, 1), length(fine_grid_a))
a_grid_r = range(gpar.a_min, gpar.a_max, gpar.N_a)

# Loop over each productivity level
for rho_i in 1:size(policy_a, 1)
    # Get the policy function for the current productivity level
    policy_values = vec(policy_l[rho_i, :])

    # Create the cubic spline interpolant
    itp = linear_interpolation(a_grid_r, policy_values, extrapolation_bc=Interpolations.Flat())

    # Evaluate the interpolant on the fine grid
    interp_policy_l[rho_i, :] = itp.(fine_grid_a)
end

# Plot
# Define colors for each productivity level
colors = palette(:viridis, size(policy_l, 1));

# Plot the interpolated policy functions
pfl_int = plot(
    fine_grid_a, interp_policy_l[1, :],
    label="ρ = $(rho_grid[1])", color=colors[1], linewidth=2,
    xlabel = "Assets (a)", ylabel = "Labor choice (ℓ)",
    title = "Policy Functions for Labor",
    legend = :bottomright);

for rho_i in 2:gpar.N_rho    plot!(pfl_int, fine_grid_a, interp_policy_l[rho_i, :],
          label="ρ = $(rho_grid[rho_i])",
          color=colors[rho_i], linewidth=2)
end

# Display and save
display(pfl_int)

# Create filename and save
# filename = "asset_policy_int_l" * "$gpar.N_l" * "_a" * "$gpar.N_a" * ".png"
# savefig(pfl_int, "output/preliminary/" * filename)

