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


#---------------------------------# A. WEALTH #--------------------------------#


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


#---------------------------------# B. INCOME #--------------------------------#


function analyze_income_dist(distYlabor_pretax::Matrix{Float64},
                                     stat_dist::Matrix{Float64};
                                     n_deciles::Int = 10,
                                     plot::Bool = true,
                                     labor_tax_policy = labor_tax_policy,
                                     analyse_tax_rates::Bool = true)

    # Flatten inputs
    incomes = vec(distYlabor_pretax)
    masses = vec(stat_dist)

    # Sort by income
    sorted_indices = sortperm(incomes)
    sorted_incomes = incomes[sorted_indices]
    sorted_masses = masses[sorted_indices]

    # Compute cumulative mass
    cum_mass = cumsum(sorted_masses)
    total_mass = cum_mass[end]

    # Compute thresholds
    thresholds = range(0, stop=sorted_incomes[end], length=n_deciles+1)
    decile_shares = Float64[]
    start_idx = 1

    for i in 1:n_deciles
        # Find upper bound of the current decile
        end_val = thresholds[i+1]
        end_idx = searchsortedlast(sorted_incomes, end_val)
        push!(decile_shares, sum(sorted_masses[start_idx:end_idx]))
        start_idx = end_idx + 1
    end

    # # Clean from zero shares
    # keep = findall(decile_shares .> 0.0)
    # decile_shares = decile_shares[keep]
    # n_deciles = length(decile_shares)

    # Plot if requested
    if plot
        fig = CairoMakie.Figure(size = (800, 500))
        ax = CairoMakie.Axis(fig[1, 1], title = "Population Share by Income Decile",
                             xlabel = "Income Decile", ylabel = "Share of Population",
                             xticks = (1:n_deciles, ["D$i" for i in 1:n_deciles]))
        CairoMakie.barplot!(ax, 1:n_deciles, decile_shares, color = :dodgerblue)
        CairoMakie.xlims!(ax, 0.5, n_deciles + 0.5)
        CairoMakie.autolimits!(ax)
        CairoMakie.display(fig)
    end

    if analyse_tax_rates
        # Return average tax rates per decile 
        labor_taxes = vec(labor_tax_policy)[sorted_indices]
        labor_taxes_collected = []
        average_rates = []
        start_idx = 1
        for i in 1:n_deciles
            # Find upper bound of the current decile
            end_val = thresholds[i+1]
            end_idx = searchsortedlast(sorted_incomes, end_val)
            average_weights = sorted_masses[start_idx:end_idx] ./ sum(sorted_masses[start_idx:end_idx])
            push!(labor_taxes_collected, sum(labor_taxes[start_idx:end_idx] .* sorted_masses[start_idx:end_idx]))
            push!(average_rates, sum((labor_taxes[start_idx:end_idx] ./ sorted_incomes[start_idx:end_idx]) .* average_weights))
            start_idx = end_idx + 1
        end
        
        return decile_shares, labor_taxes_collected, average_rates, collect(thresholds[2:end])
    else 
        return decile_shares, collect(thresholds[2:end])
    end
end


function compute_income_distribution_stats(stat_dist, distYlabor_pretax; 
    cutoffs = [0.5, (0.5, 0.9), -0.1])
    # Flatten arrays
    weights = vec(stat_dist)
    incomes = vec(distYlabor_pretax)

    # Sort by income
    sorted_idx = sortperm(incomes)
    sorted_incomes = incomes[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Normalize weights
    sorted_weights ./= sum(sorted_weights)

    # Compute cumulative shares
    cum_weights = cumsum(sorted_weights)
    cum_income = cumsum(sorted_incomes .* sorted_weights)
    aggYlabor_pretax = cum_income[end]
    # cum_income ./= aggYlabor_pretax

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
            cdf_idx = findfirst(x -> x > cutoff, cum_weights)

            # Adjust for differences - remove excess population share
            pop_res = cum_weights[cdf_idx] - cutoff

            # Compute cumulative income and share of income adjusting for residual in last element
            income_held = cum_income[cdf_idx] - sorted_incomes[cdf_idx] * pop_res # Final income
            
            # Get bottom or top share according to what sought
            income_held = c > 0 ? income_held : aggYlabor_pretax - income_held
            income_share = income_held / aggYlabor_pretax

            # Store results
            push!(dist_stats, (c, income_share, income_held))

        #-----------# Second case: range (both positive values) #-----------# 

        elseif isa(c, Tuple) && c[1] > 0 && c[2] > 0
            # Extract values from tuple
            cutoff_l = c[1]
            cutoff_h = c[2]
            # High end
            # Get cumulative mass distribution cutoff from CDF using relative bottom
            cdf_idx_h = findfirst(x -> x > cutoff_h, cum_weights)
            cdf_level_h = cum_weights[cdf_idx_h]

            # Adjust for differences - remove excess population share
            pop_res = cdf_level_h - cutoff_h

            # Compute cumulative wealth and share of wealth adjusting for residual in last element
            income_held_h = cum_income[cdf_idx_h] - sorted_incomes[cdf_idx_h] * pop_res # Final income

            # Repeat for low bottom share
            cdf_idx_l = findfirst(x -> x > cutoff_l, cum_weights)
            cdf_level_l = cum_weights[cdf_idx_l]

            # Compute cumulative wealth and share of wealth adjusting for residual in last element
            pop_res = cdf_level_l - cutoff_l
            income_held_l = cum_income[cdf_idx_l] - sorted_incomes[cdf_idx_l] * pop_res # Final income

            # Extract difference
            income_held = income_held_h - income_held_l
            income_share = income_held / aggYlabor_pretax

            # Store results
            push!(dist_stats, (c, income_share, income_held))
        
        else 
            error("Check function arguments!")
        end
    end
    return dist_stats
end

function compute_average_income_stats(stat_dist, distYlabor_pretax; 
    cutoffs = [0.5, 0.9, -0.1])
    # Flatten arrays
    weights = vec(stat_dist)
    incomes = vec(distYlabor_pretax)

    # Sort by income
    sorted_idx = sortperm(incomes)
    sorted_incomes = incomes[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Normalize weights
    sorted_weights ./= sum(sorted_weights)

    # Compute cumulative shares
    cum_weights = cumsum(sorted_weights)
    cum_income = cumsum(sorted_incomes .* sorted_weights)
    aggYlabor_pretax = cum_income[end]
    # cum_income ./= aggYlabor_pretax

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
            cdf_idx = findfirst(x -> x > cutoff, cum_weights)

            # Adjust for differences - remove excess population share
            pop_res = cum_weights[cdf_idx] - cutoff

            if c > 0 
                # Extract relevant income subdistribution, adjusting population share at cutoff
                subdist = sorted_weights[1:cdf_idx]
                subdist[end] = subdist[end] - pop_res
                subweights = subdist / sum(subdist) # Rescale weights 
                avg_income = sorted_incomes[1:cdf_idx]' * subweights # Final income
            else 
                # Extract relevant income subdistribution, adjusting population share at cutoff
                subdist = sorted_weights[cdf_idx:end]
                subdist[1] = pop_res
                subweights = subdist / sum(subdist) # Rescale weights 
                avg_income = sorted_incomes[cdf_idx:end]' * subweights # Final income
            end

            # Store results
            push!(dist_stats, (c, avg_income))
        end
    end
    return dist_stats
end


#------------------------------# C. CONSUMPTION #-----------------------------#


function analyze_consumption_dist(policy_c::Matrix{Float64},
    stat_dist::Matrix{Float64};
    n_deciles::Int = 10,
    plot::Bool = true,
    labor_tax_policy = labor_tax_policy,
    analyse_tax_rates::Bool = true)

    # Flatten inputs
    incomes = vec(distYlabor_pretax)
    masses = vec(stat_dist)

    # Sort by income
    sorted_indices = sortperm(incomes)
    sorted_incomes = incomes[sorted_indices]
    sorted_masses = masses[sorted_indices]

    # Compute cumulative mass
    cum_mass = cumsum(sorted_masses)
    total_mass = cum_mass[end]

    # Compute thresholds
    thresholds = range(0, stop=sorted_incomes[end], length=n_deciles+1)
    decile_shares = Float64[]
    start_idx = 1

    for i in 1:n_deciles
        # Find upper bound of the current decile
        end_val = thresholds[i+1]
        end_idx = searchsortedlast(sorted_incomes, end_val)
        push!(decile_shares, sum(sorted_masses[start_idx:end_idx]))
        start_idx = end_idx + 1
    end

    # # Clean from zero shares
    # keep = findall(decile_shares .> 0.0)
    # decile_shares = decile_shares[keep]
    # n_deciles = length(decile_shares)

    # Plot if requested
    if plot
        fig = CairoMakie.Figure(size = (800, 500))
        ax = CairoMakie.Axis(fig[1, 1], title = "Population Share by Income Decile",
        xlabel = "Income Decile", ylabel = "Share of Population",
        xticks = (1:n_deciles, ["D$i" for i in 1:n_deciles]))
        CairoMakie.barplot!(ax, 1:n_deciles, decile_shares, color = :dodgerblue)
        CairoMakie.xlims!(ax, 0.5, n_deciles + 0.5)
        CairoMakie.autolimits!(ax)
        CairoMakie.display(fig)
    end

    if analyse_tax_rates
    # Return average tax rates per decile 
        labor_taxes = vec(labor_tax_policy)[sorted_indices]
        labor_taxes_collected = []
        average_rates = []
        start_idx = 1
        for i in 1:n_deciles
            # Find upper bound of the current decile
            end_val = thresholds[i+1]
            end_idx = searchsortedlast(sorted_incomes, end_val)
            average_weights = sorted_masses[start_idx:end_idx] ./ sum(sorted_masses[start_idx:end_idx])
            push!(labor_taxes_collected, sum(labor_taxes[start_idx:end_idx] .* sorted_masses[start_idx:end_idx]))
            push!(average_rates, sum((labor_taxes[start_idx:end_idx] ./ sorted_incomes[start_idx:end_idx]) .* average_weights))
            start_idx = end_idx + 1
        end

        return decile_shares, labor_taxes_collected, average_rates, collect(thresholds[2:end])
    else 
        return decile_shares, collect(thresholds[2:end])
    end
end


#--------------------------------# D. TAXATION #-------------------------------#


function compute_average_rate_stats(stat_dist, labor_tax_policy;
                                    distYlabor_pretax = distYlabor_pretax, 
                                    cutoffs = [1, 0.5, 0.9, -0.1])
    # Flatten arrays
    weights = vec(stat_dist)
    taxes = vec(labor_tax_policy)
    labor_tax_rate_policy = vec(labor_tax_policy ./ distYlabor_pretax)

    # Sort by income
    sorted_idx = sortperm(taxes)
    sorted_taxes = taxes[sorted_idx]
    sorted_weights = weights[sorted_idx]
    sorted_rates = labor_tax_rate_policy[sorted_idx]

    # Normalize weights
    sorted_weights ./= sum(sorted_weights)

    # Compute cumulative shares
    cum_weights = cumsum(sorted_weights)

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
            cdf_idx = findfirst(x -> x > cutoff, cum_weights)

            # Adjust for differences - remove excess population share
            pop_res = cum_weights[cdf_idx] - cutoff

            if c > 0 
                # Extract relevant income subdistribution, adjusting population share at cutoff
                subdist = sorted_weights[1:cdf_idx]
                subdist[end] = subdist[end] - pop_res
                subweights = subdist / sum(subdist) # Rescale weights 
                avg_rate = sorted_rates[1:cdf_idx]' * subweights # Final income
            else 
                # Extract relevant income subdistribution, adjusting population share at cutoff
                subdist = sorted_weights[cdf_idx:end]
                subdist[1] = pop_res
                subweights = subdist / sum(subdist) # Rescale weights 
                avg_rate = sorted_rates[cdf_idx:end]' * subweights # Final income
            end

            # Store results
            push!(dist_stats, (c, avg_rate))
        end
    end
    return dist_stats
end