###############################################################################
############################## MODEL_SOLUTION.JL ##############################

############################# This script solves ##############################
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

using LinearAlgebra
using Distances
using Base.Threads

include("model_parameters.jl")
include("production_and_government.jl")
include("auxiliary_functions.jl")
include("numerics.jl")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 1. INITIALIZE GRIDS FOR OPTIMISATION  #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Assets
a_grid = makeGrid(a_min, a_max, N_a)

# Labor
l_grid = makeGrid(l_min, l_max, N_l)

# Labor productivity - Defined in model_parameters.jl
# rho_grid = rho_grid

# Taxation parameters


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 2. VALUE FUNCTION ITERATION #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

w = 1
r = 0.05
# VFI = function(a_grid, rho_grid, l_grid, V0, pi_rho, w, Taxes)

    #Initialise utility function
    U = zeros(N_rho, N_a, N_l, N_a) # Productivity X Assets t X Labor choice X Assets t+1

    #Compute total labor income for each combination of labor and productivity
    y = (l_grid * rho_grid') .* w 

    # Compute taxation for each degree of progressivity - TBM: can be wrapped in a function
    # Expanding dimensions for broadcasting
    reshaped_y = ExpandMatrix(y, taxes.N_tau_y)
    Tau_y = Vector2NDimMatrix(taxes.tau_y, ndims(y))

    # Compute the tax-adjusted income matrix - X: labor, Y: productivity, Z: progressivity degree
    T_y = reshaped_y .* ones(1, 1, taxes.N_tau_y) .- taxes.lambda_y .* reshaped_y .^ (1 .- Tau_y);

    # Compute disposable income after asset transfers (savings a' and interests (1+r)a)
    # Disposable income for each possible asset-state-specific interests yielded from t-1 
    # 4th dim: a
    y_after_interests = ExpandMatrix(T_y, N_a)
    interests = Vector2NDimMatrix((1 + r) .* a_grid, ndims(T_y))

    y_after_interests = y_after_interests .+ interests;

    # Find resource allocated to consumption (consumption + consumption taxes) for each combination of 
    # labor, productivity, degree of labor income progressivity, assets today
    # 5th dim: a_prime 

    consumption_expenditure = ExpandMatrix(y_after_interests, N_a)
    savings = Vector2NDimMatrix(a_grid, ndims(y_after_interests))

    consumption_expenditure = consumption_expenditure .- savings;

    # Disentangle consumption from consumption + consumption taxes (Feldstein specification)
    # for each degree of consumption tax progressivity

    hh_consumption = ExpandMatrix(consumption_expenditure, taxes.N_tau_c);
    # Vectorisation attempt
    # tau_c_reshaped = Vector2NDimMatrix(taxes.tau_c, ndims(consumption_expenditure))

    # consumption = find_c_feldstein.(consumption_expenditure, taxes.lambda_c, tau_c_reshaped)
    size(hh_consumption)

    hh_consumption .= max.(hh_consumption, 0);

    @time @threads for i in 1:taxes.N_tau_c
        # Set progressivity rate
        prog_rate = taxes.tau_c[i]
        # Find consumption
        hh_consumption[:, :, :, :, :, i] = find_c_feldstein.(hh_consumption[:, :, :, :, :, i], 
                                                            taxes.lambda_c, prog_rate) 
    end

    # Check
    hh_consumption[1, 1, 1, 1, 1, :]

    # TO DO: 
    # 1. MANUAL CHECK OF RESULTS
    # 2. CHECK PRE-EXISTING ZEROS AND ADJUST NEGATIVE VALUES IF NEEDED  
    #Adjust negative values



    # Find consumption for each degree of 
    for l in 1:N_l
        for rho in 1:N_rho
            for tau in 1:taxes.N_tau_y
                for a in 1:N_a
                    for a_prime in 1:N_a
                    # Extract income
                    y



VFI = function(r, w, agrid, sgrid, V0, prob, par)

    Ns = length(sgrid)
    Na = length(agrid)


    U = zeros(Ns, Na, Na)

    for is in 1:Ns                     # Loop Over skills Today
        for ia in 1:Na                 # Loop Over assets Today
            for ia_p in 1:Na           # Loop Over assets Tomorrow
                s = sgrid[is]         # Technology Today
                a = agrid[ia]         # Capital Today
                a_p = agrid[ia_p]     # Capital Tomorrow
                # Solve for Consumption at Each Point
                c = (1 + r) * a + s * w - a_p
                if c .< 0
                    U[is, ia, ia_p] = -10^6
                else
                    ()
                    U[is, ia, ia_p] = c^(1 - par.mu) / (1 - par.mu)
                end
            end
        end
    end
