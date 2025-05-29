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
#--------------------# 1. GENERIC DISTRIBUTION FUNCTIONS #--------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function build_density_dict(df::DataFrame, dist_sym::Symbol, idx_low::Int, idx_mid::Int, idx_high::Int)
    # Returns density for low, middle and high scenario for selected distribution
    # Extract 3D distribution from the DataFrame
    dist = df[!, dist_sym]

    # Pull out the three selected slices
    dist_low = dist[idx_low]
    dist_mid = dist[idx_mid]
    dist_high = dist[idx_high]

    # Aggregate over productivity (dim = 1), normalize to get densities over wealth
    dens_low = vec(sum(dist_low, dims = 1) ./ sum(dist_low))
    dens_mid = vec(sum(dist_mid, dims = 1) ./ sum(dist_mid))
    dens_high = vec(sum(dist_high, dims = 1) ./ sum(dist_high))

    return Dict(:low => dens_low, :mid => dens_mid, :high => dens_high)
end

function extract_low_mid_high(df, col_sym;
                              idx_low = idx_low_tau_y,
                              idx_mid = idx_middle,
                              idx_high = idx_high_tau_y,
                              table_string = nothing,
                              as_percentage = false,
                              normalise = "no"
                              )
    # This function extracts low, mid and high scenario values for selected aggregate
    # And outputs it to add it to the dataframe collecting aggregate results
    # Extract column from the DataFrame
    agg = df[!, col_sym]

    # Pull out the three selected values
    col_low = agg[idx_low]
    col_mid = agg[idx_mid]
    col_high = agg[idx_high]

    # Normalise if desired
    if normalise == "num"
        col_low = col_low / col_mid
        col_high = col_high / col_mid
        col_mid = col_mid / col_mid
    elseif normalise == "var"
        col_low = col_low / col_mid - 1
        col_high = col_high / col_mid - 1
        col_mid = col_mid / col_mid - 1
    else
        nothing 
    end

    # Transform into percentages and round if desired
    if as_percentage
        if isa(col_low, Number)
            col_low = round(col_low * 100, digits = 2)
            col_mid = round(col_mid * 100, digits = 2)
            col_high = round(col_high * 100, digits = 2)
        elseif isa(col_low, AbstractArray) || isa(col_low, Tuple)
            col_low = round.(col_low .* 100, digits = 2)
            col_mid = round.(col_mid .* 100, digits = 2)
            col_high = round.(col_high .* 100, digits = 2)
        end
    end

    # Create vector 
    table_string = isnothing(table_string) ? string(col_sym) : table_string
    table_row = [table_string, col_low, col_mid, col_high]

    return table_row
end

function extract_acp_vs_rob(df, rowname;
                              var_col = "variable",
                              baseline_col = "b",
                              acp_col = "acp",
                              r1_col = "lc_eq",
                              r2_col = "Crev_eq",
                              table_string = nothing,
                              as_percentage = false,
                              normalise = "no",
                              return_baseline = true
                              )
    # This function extracts low, mid and high scenario values for selected aggregate
    # And outputs it to add it to the dataframe collecting aggregate results
    # Extract row from the DataFrame
    agg = df[df[:, var_col] .== rowname, :]

    # Get target cols
    target_cols = setdiff(names(df), [var_col, baseline_col])
    num_cols = setdiff(names(df), [var_col])

    # Normalise if desired
    if normalise == "num"
        base = agg[1, baseline_col]
        for col in target_cols
            agg[1, col] = round(agg[1, col] / base, digits = 4)
        end
    elseif normalise == "var"
        base = agg[1, baseline_col]
        for col in target_cols
            agg[1, col] = round((agg[1, col] - base) / base, digits = 4)
            print(agg[1, col])
        end
    elseif normalise == "no"
        nothing
    else
        error("Check your -normalise attribute!") 
    end

    # Transform into percentages and round if desired
    if as_percentage
        if isa(agg[1, baseline_col], Number)
            for col in num_cols
                agg[1, col] = agg[1, col] * 100
            end
        elseif isa(agg[1, baseline_col], AbstractArray) || isa(agg[1, baseline_col], Tuple)
            for col in num_cols
                agg[1, col] = agg[1, col] .* 100
            end
        end
    end

    # Replace table name 
    if !isnothing(table_string)
        agg[1, var_col] = table_string
    end

    # Remove baseline if asked 
    if !return_baseline
        select!(agg, Not(baseline_col))
    end

    return agg
end

function compute_decile_distribution(stat_dist::AbstractMatrix, target_dist::AbstractMatrix)
    # Flatten
    weights = vec(stat_dist)
    values = vec(target_dist)

    # Filter out invalid entries
    valid = .!isnan.(weights) .& .!isnan.(values)
    weights = weights[valid]
    values = values[valid]

    # Income proxy for sorting: weighted values
    sort_idx = sortperm(values)
    sorted_weights = weights[sort_idx]
    sorted_values = values[sort_idx]

    sorted_weights ./= sum(sorted_weights)
    cum_weights = cumsum(sorted_weights)

    # Replace with 1 last element to avoid rounding issues
    cum_weights[end] = 1.0

    # Decile thresholds
    decile_bounds = [0.1i for i in 1:10]
    decile_shares = Float64[]

    start_idx = 1
    for bound in decile_bounds
        stop_idx = findfirst(x -> x >= bound, cum_weights)
        overflow = cum_weights[stop_idx] - bound

        w_sub = sorted_weights[start_idx:stop_idx]
        v_sub = sorted_values[start_idx:stop_idx]
        w_sub[end] -= overflow

        share = sum(v_sub .* w_sub) / sum(values .* weights)
        push!(decile_shares, share)

        start_idx = stop_idx + 1
    end

    return decile_shares
end

function compute_avg_cevs_per_income_decile(cev_matrix::Matrix, policy_l_ptinc::Matrix, stat_dist::Matrix)
    # Flatten
    cev = vec(cev_matrix)
    income = vec(policy_l_ptinc)
    weights = vec(stat_dist)

    # Clean
    valid = .!isnan.(cev) .& .!isnan.(income) .& .!isnan.(weights)
    cev = cev[valid]
    income = income[valid]
    weights = weights[valid]

    # Normalize weights
    weights ./= sum(weights)

    # Sort by income
    sort_idx = sortperm(income)
    cev = cev[sort_idx]
    weights = weights[sort_idx]

    # Compute cumulative population weights
    cum_weights = cumsum(weights)

    # Adjust last element to avoid approx errors
    if abs(cum_weights[end] - 1) < 10^(-6)
        cum_weights[end] = 1
    end

    # Compute decile cutoffs
    deciles = [0.1i for i in 1:10]
    avg_cevs = Float64[]

    start_idx = 1
    for d in deciles
        stop_idx = findfirst(x -> x >= d, cum_weights)
        pop_over = cum_weights[stop_idx] - d

        w_sub = weights[start_idx:stop_idx]
        c_sub = cev[start_idx:stop_idx]
        w_sub[end] -= pop_over
        w_sub ./= sum(w_sub)

        push!(avg_cevs, sum(c_sub .* w_sub))

        start_idx = stop_idx + 1
    end

    return avg_cevs
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------------# 2. SPECIFIC DISTRIBUTIONS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# A. WEALTH #-------------------------------#
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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# B. INCOME #--------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function analyze_income_dist(policy_l_incpt::Matrix{Float64},
                                     stat_dist::Matrix{Float64};
                                     n_deciles::Int = 10,
                                     plot::Bool = true,
                                     labor_tax_policy = labor_tax_policy,
                                     analyse_tax_rates::Bool = true)

    # Flatten inputs
    incomes = vec(policy_l_incpt)
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


function compute_income_distribution_stats(stat_dist, policy_l_incpt; 
    cutoffs = [0.5, (0.5, 0.9), -0.1])
    # Flatten arrays
    pop_weights = vec(stat_dist)
    incomes = vec(policy_l_incpt)

    # Sort by income
    sorted_idx = sortperm(incomes)
    sorted_incomes = incomes[sorted_idx]
    sorted_weights = pop_weights[sorted_idx]

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

function compute_average_income_stats(stat_dist, policy_l_incpt; 
    cutoffs = [0.5, 0.9, -0.1])
    # This function returns average income per population group
    # Flatten arrays
    weights = vec(stat_dist)
    incomes = vec(policy_l_incpt)

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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------------------# C. CONSUMPTION #-----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function analyze_consumption_dist(policy_c::Matrix{Float64},
    stat_dist::Matrix{Float64};
    n_deciles::Int = 10,
    plot::Bool = true,
    labor_tax_policy = labor_tax_policy,
    analyse_tax_rates::Bool = true)

    # Flatten inputs
    incomes = vec(policy_l_incpt)
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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------------------# D. TAXATION #-------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

function compute_average_rate_stats(stat_dist, labor_tax_policy;
                                    policy_l_incpt = policy_l_incpt, 
                                    cutoffs = [0.5, 0.9, -0.1])
    # Flatten arrays
    pop_weights = vec(stat_dist)
    taxes = vec(labor_tax_policy)
    labor_tax_rate_policy = vec(labor_tax_policy ./ policy_l_incpt)

    # Filter out NaNs from all vectors
    valid = .!isnan.(pop_weights) .& .!isnan.(taxes) .& .!isnan.(labor_tax_rate_policy)

    pop_weights = pop_weights[valid]
    taxes = taxes[valid]
    labor_tax_rate_policy = labor_tax_rate_policy[valid]

    # Sort by income
    sorted_idx = sortperm(taxes)
    sorted_taxes = taxes[sorted_idx]
    sorted_weights = pop_weights[sorted_idx]
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

#----# Average labor income tax effective rate distribution #----#
function compute_decile_average_rates(stat_dist, labor_tax_policy;
    policy_l_incpt = policy_l_incpt,
    bar_labels = true,
    save_plot = false,
    save_path = "output/figures/baseline/labor_tax_aer_deciles.png",
    cmap = :Spectral_10,
    title_str = "Average Effective Labor Tax Rate by Income Decile",
    labx = "Income Decile",
    laby = "Avg. Tax Rate",
    as_percentage = false
)
    # Flatten arrays
    pop_weights = vec(stat_dist)
    taxes = vec(labor_tax_policy)
    labor_tax_rate_policy = vec(labor_tax_policy ./ policy_l_incpt)

    # Remove NaNs
    valid = .!isnan.(pop_weights) .& .!isnan.(taxes) .& .!isnan.(labor_tax_rate_policy)
    pop_weights = pop_weights[valid]
    taxes = taxes[valid]
    labor_tax_rate_policy = labor_tax_rate_policy[valid]

    # Sort by income (taxes used as income proxy)
    sorted_idx = sortperm(taxes)
    sorted_weights = pop_weights[sorted_idx]
    sorted_rates = labor_tax_rate_policy[sorted_idx]

    # Normalize weights
    sorted_weights ./= sum(sorted_weights)

    # Compute cumulative weights
    cum_weights = cumsum(sorted_weights)

    # Decile cutoffs
    deciles = [0.1i for i in 1:10]
    decile_avgrates = Float64[]

    start_idx = 1
    for d in deciles
        # Find upper bound index
        stop_idx = findfirst(x -> x >= d, cum_weights)
        # Adjust for over-cutoff
        pop_res = cum_weights[stop_idx] - d

        subdist = sorted_weights[start_idx:stop_idx]
        subdist[end] -= pop_res
        subweights = subdist ./ sum(subdist)
        avg_rate = sorted_rates[start_idx:stop_idx]' * subweights

        push!(decile_avgrates, avg_rate)
        start_idx = stop_idx + 1
    end

    if as_percentage
        decile_avgrates = decile_avgrates * 100
    end 

    # === Plotting ===
    fig = CairoMakie.Figure(size = (800, 500))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = labx,
        ylabel = laby,
        title = title_str,
        titlesize = 20,
        xlabelsize = 14,
        ylabelsize = 14,
        xticks = (1:10, ["D$i" for i in 1:10])
    )

    colors = ColorSchemes.colorschemes[cmap].colors[1:10]
    CairoMakie.barplot!(ax, 1:10, decile_avgrates; color = colors)

    # CairoMakie.axislegend(ax, position = :rt)

    if bar_labels
        for (i, val) in enumerate(decile_avgrates)
            offset = val ≥ 0 ? 0.01 : -0.01
            valign = val ≥ 0 ? :bottom : :top

            CairoMakie.text!(ax, i, val + offset;
                text = string(round(val, digits = as_percentage ? 1 : 2)),
                align = (:center, valign),
                fontsize = 12)
        end
    end

    if save_plot
        CairoMakie.save(save_path, fig)
    end

    return decile_avgrates, fig
end


#----# Average labor income tax marginal rate distribution #----#
function compute_decile_marginal_rates(stat_dist, policy_l_incpt, lambda_y, tau_y;
    bar_labels = true,
    save_plot = false,
    save_path = "output/figures/labor_mtr_deciles.png",
    cmap = :PiYG_10,
    eps = 1e-4,
    title_str = "Average Effective Labor Tax Rate by Income Decile",
    labx = "Income Decile",
    laby = "Avg. Tax Rate",
    as_percentage = false
)
    # Original income and taxes
    inc = vec(policy_l_incpt)
    tax = inc .- lambda_y .* inc .^ (1 - tau_y)

    # Perturb income slightly
    inc_up = inc .+ eps
    tax_up = inc_up .- lambda_y .* inc_up .^ (1 - tau_y)

    # Marginal rate
    marginal_rate = (tax_up .- tax) ./ eps

    # Flatten weights
    weights = vec(stat_dist)

    # Filter out NaNs
    valid = .!isnan.(weights) .& .!isnan.(marginal_rate)
    weights = weights[valid]
    marginal_rate = marginal_rate[valid]
    inc = inc[valid]

    # Sort by income
    sorted_idx = sortperm(inc)
    sorted_weights = weights[sorted_idx]
    sorted_mtr = marginal_rate[sorted_idx]

    sorted_weights ./= sum(sorted_weights)
    cum_weights = cumsum(sorted_weights)

    # Compute by decile
    deciles = [0.1i for i in 1:10]
    decile_mtr = Float64[]

    start_idx = 1
    for d in deciles
        stop_idx = findfirst(x -> x >= d, cum_weights)
        pop_res = cum_weights[stop_idx] - d

        subdist = sorted_weights[start_idx:stop_idx]
        subdist[end] -= pop_res
        subweights = subdist ./ sum(subdist)

        avg_rate = sorted_mtr[start_idx:stop_idx]' * subweights
        push!(decile_mtr, avg_rate)

        start_idx = stop_idx + 1
    end

    if as_percentage
        decile_mtr = decile_mtr * 100
    end

    # === Plotting ===
    fig = CairoMakie.Figure(size = (800, 500))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = labx,
        ylabel = laby,
        title = title_str,
        titlesize = 20,
        xlabelsize = 14,
        ylabelsize = 14,
        xticks = (1:10, ["D$i" for i in 1:10])
    )

    colors = ColorSchemes.colorschemes[cmap].colors[1:10]
    CairoMakie.barplot!(ax, 1:10, decile_mtr; color = colors)

    if bar_labels
        for (i, val) in enumerate(decile_mtr)
            CairoMakie.text!(ax, i, val + 0.01;  # small offset above bar
                text = string(round(val, digits= as_percentage ? 1 : 2)),
                align = (:center, :bottom),
                fontsize = 12)
        end
    end

    if save_plot
        CairoMakie.save(save_path, fig)
    end

    return decile_mtr, fig
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#--------------------------------# D. WELFARE #-------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Following Ferriere et al. (2023), compute Consumption Equivalent Variation

function compute_consumption_value(policy_c, policy_a, pi_rho; 
                                   a_grid = a_grid, gpar = gpar, 
                                   beta = hhpar.beta, rra = hhpar.rra, 
                                   tol=1e-10, max_iter=10_000)
    """
    Computes C(a,z): the expected discounted utility from consumption only,
    using policy_c, policy_a and the exogenous transition matrix pi.
    """
    N_rho, N_a = gpar.N_rho, gpar.N_a
    C = zeros(N_rho, N_a)
    C_new = similar(C)

    for iter in 1:max_iter
        for a in 1:N_a, z in 1:N_rho
            u_c = (policy_c[z, a])^(1 - rra) / (1 - rra)
            cont = 0.0

            a_prime = policy_a[z, a]
            if a_prime <= a_grid[1]
                a_i_low, a_i_high, w_high = 1, 1, 1.0
            elseif a_prime >= a_grid[end]
                a_i_low, a_i_high, w_high = N_a, N_a, 1.0
            else
                a_i_high = searchsortedfirst(a_grid, a_prime)
                a_i_low = a_i_high - 1
                w_high = (a_prime - a_grid[a_i_low]) / (a_grid[a_i_high] - a_grid[a_i_low])
            end

            for z_next in 1:N_rho
                prob = pi_rho[z, z_next]
                cont += prob * (
                    (1 - w_high) * C[z_next, a_i_low] +
                    w_high * C[z_next, a_i_high]
                )
            end

            C_new[z, a] = u_c + beta * cont
        end

        if maximum(abs.(C_new .- C)) < tol
            return C_new
        end
        C .= C_new
    end

    error("compute_consumption_value did not converge")
end

function recompute_value(policy_c, policy_l, policy_a, pi_rho;
                         a_grid, gpar, hhpar, tol=1e-10, max_iter=10_000)
    N_rho, N_a = gpar.N_rho, gpar.N_a
    V = zeros(N_rho, N_a)
    V_new = similar(V)

    for iter in 1:max_iter
        for a in 1:N_a, z in 1:N_rho
            u = get_utility_hh(policy_c[z, a], policy_l[z, a], hhpar)
            cont = 0.0

            a_prime = policy_a[z, a]
            if a_prime <= a_grid[1]
                a_i_low, a_i_high, w_high = 1, 1, 1.0
            elseif a_prime >= a_grid[end]
                a_i_low, a_i_high, w_high = N_a, N_a, 1.0
            else
                a_i_high = searchsortedfirst(a_grid, a_prime)
                a_i_low = a_i_high - 1
                w_high = (a_prime - a_grid[a_i_low]) / (a_grid[a_i_high] - a_grid[a_i_low])
            end

            for z_next in 1:N_rho
                prob = pi_rho[z, z_next]
                cont += prob * (
                    (1 - w_high) * V[z_next, a_i_low] +
                    w_high * V[z_next, a_i_high]
                )
            end

            V_new[z, a] = get_utility_hh(policy_c[z, a], policy_l[z, a], hhpar) + hhpar.beta * cont
        end

        if maximum(abs.(V_new .- V)) < tol
            return V_new
        end
        V .= V_new
    end

    error("recompute_value did not converge")
end


function compute_cev(v_reform, v_base, policy_c, policy_a, stat_dist, pi_rho;
                     a_grid = a_grid, gpar = gpar, 
                     beta = hhpar.beta, rra = hhpar.rra, 
                     tol=1e-10, max_iter=10_000,
                     as_percentage = true)
    """
    Computes individual and aggregate consumption equivalent variation (CEV).
    """
    C = compute_consumption_value(policy_c, policy_a, pi_rho; 
                                   a_grid = a_grid, gpar = gpar, 
                                   beta = hhpar.beta, rra = hhpar.rra, 
                                   tol=1e-10, max_iter=10_000)
    
    cev = -1 .+ (1 .+ (v_reform .- v_base) ./ C).^(1 / (1 - rra))
    aggCEV = sum(stat_dist .* cev)

    if as_percentage
        cev = cev .* 100
        aggCEV = aggCEV * 100
    end

    return cev, aggCEV
end