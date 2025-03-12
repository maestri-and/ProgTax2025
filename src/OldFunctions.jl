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



###### COMPUTING CONSUMPTION, TAXES AND UTILITY ######

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
