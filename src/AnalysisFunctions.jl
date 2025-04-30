###############################################################################
############################# ANALYSISFUNCTIONS.JL ############################

##### This script collects functions used in the analysis of the results ######
######## produced by simulations of the benchmark ProgTax(2025) model #########

###############################################################################

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------------# 1. DISTRIBUTIONS #-----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function compute_wealth_distribution_stats(stat_dist, a_grid; 
    cutoffs = [0.5, (0.3, 0.7), -0.1], replace_debt = false)
    # Flatten stationary distribution and get cumulative distribution
    distA = vec(sum(stat_dist, dims = 1))
    cdfA = cumsum(distA, dims = 1)

    # Remove debt, consider only positive wealth 
    a_grid = replace_debt ? max.(a_grid, 0.0) : a_grid

    # Plot cumulative distribution
    Plots.plot(cdfA,)

    # Compute total assets in the economy 
    aggA = distA' * a_grid
    
    # Initialise list of results
    dist_stats = []

    for c in cutoffs
    # Three types of arguments can be passed into cutoff as shares
    # Positive float x => Bottom x share  
    # Negative float x => Top x share
    # Tuple of two floats x, y => Middle share from x to y 
        #-----------# First and second case: bottom and top share #-----------# 
        if isa(c, AbstractFloat) && ((c > 0 && c < 1) || (c < 0 && c > -1))
            # Get bottom share as cutoff, also for top share
            cutoff = c > 0 ? c : (1 + c)

            # Get cumulative mass distribution cutoff from CDF 
            cdf_idx = findfirst(x -> x > cutoff, cdfA)

            # Adjust for differences - remove excess population share
            res = cdfA[cdf_idx] - cutoff

            # Compute cumulative wealth and share of wealth adjusting for residual in last element
            subdist = distA[1:cdf_idx]
            subdist[end] = subdist[end] - res

            # Get bottom or top share according to what sought
            wealth_held = c > 0 ? subdist' * a_grid[1:cdf_idx] : aggA - subdist' * a_grid[1:cdf_idx]
            wealth_share = wealth_held / aggA

            # Store results
            push!(dist_stats, (c, wealth_share, wealth_held))

        #-----------# Second case: range (both positive values) #-----------# 

        elseif isa(c, Tuple) && c[1] > 0 && c[2] > 0
            # Extract values from tuple
            cutoff_l = c[1]
            cutoff_h = c[2]
            # High end
            # Get cumulative mass distribution cutoff from CDF using relative bottom
            cdf_idx_h = findfirst(x -> x > cutoff_h, cdfA)
            cdf_level_h = cdfA[cdf_idx_h]

            # Adjust for differences - remove excess population share
            res = cdf_level_h - cutoff_h

            # Compute cumulative wealth and share of wealth adjusting for residual in last element
            subdist_h = distA[1:cdf_idx_h]
            subdist_h[end] = subdist_h[end] - res

            wealth_held_h = subdist_h' * a_grid[1:cdf_idx_h]

            # Repeat for low bottom share
            cdf_idx_l = findfirst(x -> x > cutoff_l, cdfA)
            cdf_level_l = cdfA[cdf_idx_l]

            # Compute cumulative wealth and share of wealth adjusting for residual in last element
            res = cdf_level_l - cutoff_l
            subdist_l = distA[1:cdf_idx_l]
            subdist_l[end] = subdist_l[end] - res

            wealth_held_l = subdist_l' * a_grid[1:cdf_idx_l]

            # Extract difference
            wealth_held = wealth_held_h - wealth_held_l
            wealth_share = wealth_held / aggA

            # Store results
            push!(dist_stats, (c, wealth_share, wealth_held))
        
        else 
            error("Check function arguments!")
        end
    end
    return dist_stats
end


"""
    gini(stat_dist::Matrix, distY::Matrix; plot_curve::Bool=false)

Compute Gini coefficient and optionally plot Lorenz curve using CairoMakie.
"""
function compute_gini(stat_dist::Matrix, distY::Matrix; plot_curve::Bool=false)
    # Flatten arrays
    weights = vec(stat_dist)
    incomes = vec(distY)

    # Sort by income
    sorted_idx = sortperm(incomes)
    sorted_incomes = incomes[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Normalize weights
    sorted_weights ./= sum(sorted_weights)

    # Compute cumulative shares
    cum_weights = cumsum(sorted_weights)
    cum_income = cumsum(sorted_incomes .* sorted_weights)
    cum_income ./= cum_income[end]

    # Gini coefficient (trapezoidal rule)
    gini_val = 1 - 2 * sum(diff(cum_weights) .* (cum_income[1:end-1] + cum_income[2:end])) # trapezoid integration

    if plot_curve
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1], title = "Lorenz Curve (Gini = $(round(gini_val, digits=3)))",
                  xlabel = "Cumulative Population Share", ylabel = "Cumulative Income Share")
        
        CairoMakie.lines!(ax, cum_weights, cum_income, label="Lorenz Curve", color=:blue)
        CairoMakie.lines!(ax, [0, 1], [0, 1], linestyle=:dash, color=:black, label="Line of Equality")
        CairoMakie.axislegend(ax, position = :lt)
        display(fig)
    end

    return gini_val
end
