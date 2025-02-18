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