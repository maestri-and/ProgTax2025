###############################################################################
############################ PLOTTINGFUNCTIONS.JL #############################

###### This script defines useful plotting functions for visual insights ######
################### from the benchmark ProgTax(2025) model ####################

###############################################################################


using StatsBase
using Plots

############################## PRELIMINARY PLOTS ##############################



#-#-#-#-#-#-#                1. PLOTTING FELDSTEIN                #-#-#-#-#-#-#

# Plot for single rate of progressivity
# Flatten matrices into column vectors

function plot_taxes_vs_base(base_vec, tax_vec; base_range=nothing)
    if isnothing(base_range)
    # Sample a subset of data points for plotting
    sample_indices = sample(1:length(base_vec), min(10000, length(base_vec)), replace=false)

    # Plot the full dataset
    p = scatter(base_vec[sample_indices], tax_vec[sample_indices],
                xlabel="Tax Base", ylabel="Tax",
                title="Taxes vs Tax Base", legend=false, markercolor=:green)
    else
        # Filter the data based on the specified base range
        lower_bound, upper_bound = base_range
        filter_indices = findall(x -> lower_bound < x < upper_bound, base_vec)

        # Plot the filtered dataset
        p = scatter(base_vec[filter_indices], tax_vec[filter_indices],
                    xlabel="Tax Base", ylabel="Tax",
                    title="Taxes vs Tax Base \n Range-Adjusted", legend=false, markercolor=:red)
    end

    # Add a horizontal line at tax = 0 for reference
    hline!([0], linestyle=:dash, color=:black, lw=2, label="Tax = 0")

    return p
end


# Generic function to plot taxes vs tax base
function plot_taxes_vs_base(base_vec, tax_vec; base_range=nothing)
    if isnothing(base_range)
        # Sample a subset of data points for plotting
        sample_indices = sample(1:length(base_vec), min(10000, length(base_vec)), replace=false)

        # Plot the full dataset
        p = scatter(base_vec[sample_indices], tax_vec[sample_indices],
                    xlabel="Tax Base", ylabel="Tax",
                    title="Taxes vs Tax Base", legend=false, markercolor=:green)
    else
        # Filter the data based on the specified base range
        lower_bound, upper_bound = base_range
        filter_indices = findall(x -> lower_bound < x < upper_bound, base_vec)

        # Plot the filtered dataset
        p = scatter(base_vec[filter_indices], tax_vec[filter_indices],
                    xlabel="Tax Base", ylabel="Tax",
                    title="Taxes vs Tax Base \n Range-Adjusted", legend=false, markercolor=:red)
    end

    # Add a horizontal line at tax = 0 for reference
    hline!(p, [0], linestyle=:dash, color=:black, lw=2, label="Tax = 0")

    # Display the plot
    display(p)

    return p
end

# plot_taxes_vs_base(vec(hh_consumption), vec(hh_consumption_tax); base_range = (-0.1, 0.5))

# function plot_1D_itp_vs_data(itp, x_data, y_data; x_range = nothing, y_range = nothing)
#     if x_range


#-#-#-#-#-#-#          2. PLOTTING INTERPOLATIONS VS DATA         #-#-#-#-#-#-#

# Generic function to plot interpolation vs data
function plot_1D_itp_vs_data(itp, x_data, y_data; x_range=nothing, y_range=nothing)
    # Determine the range of x values to focus on
    if x_range === nothing
        x_min = minimum(x_data)
        x_max = maximum(x_data)
    else
        x_min, x_max = x_range
    end

    # Find indices within the specified x range
    in_range = findall(x -> x_min <= x <= x_max, x_data)

    # Subset the data within the specified range
    x_subset = x_data[in_range]
    y_subset = y_data[in_range]

    # Create a finer grid of x values for interpolation
    xq = range(x_min, stop=x_max, length=1000)

    # Sample a subset of data points for plotting
    idx = sample(1:length(x_subset), min(100, length(x_subset)), replace=false)

    # Plot the data points
    p = plot(x_subset[idx], y_subset[idx],
         seriestype=:scatter, label="Data", legend=:topleft, framestyle=:box,
         xlabel="x", ylabel="y", title="Interpolation vs Data")

    # Plot the interpolation
    plot!(p, xq, itp.(xq), label="Interpolation", linewidth=2)

    # Optionally adjust y-axis range
    if y_range !== nothing
        ylims!(y_range)
    end

    return p
end

# plot_1D_itp_vs_data(itp, x_data, y_data; x_range = (-0.0001, 0.0005))


# Generic function to plot 2D interpolated values and knots for multiple fixed levels of the first dimension
function plot_2d_policy_int_vs_data(policy_matrix, policy_spline, a_grid, rho_grid; policy_type="assets")
    """
    Plots the policy function for each productivity level, comparing raw data with Spline2D interpolation.

    Args:
        policy_matrix : Matrix (N_rho × N_a) containing raw policy function values.
        policy_spline : Spline2D object interpolating the policy function.
        a_grid        : Vector of asset grid values.
        rho_grid      : Vector of productivity grid values.
        policy_type   : String, either "assets", "labor", or "consumption" (determines labels).

    Returns:
        Displays the comparison plot for each level of ρ.
    """

    # Define labels based on the selected policy function
    if policy_type == "assets"
        title_text = "Policy Function for Assets"
        ylabel_text = "Future Assets (a')"
    elseif policy_type == "labor"
        title_text = "Policy Function for Labor"
        ylabel_text = "Labor Supplied (l)"
    elseif policy_type == "consumption"
        title_text = "Policy Function for Consumption"
        ylabel_text = "Consumption (c)"
    else
        error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
    end

    # Loop over each productivity level ρ and create a plot
    for i_rho in 1:length(rho_grid)
        rho_val = rho_grid[i_rho]

        # Create a denser asset grid for interpolation
        a_dense = range(minimum(a_grid), maximum(a_grid), length=200)
        policy_spline_values = [policy_spline(rho_val, a) for a in a_dense]

        # Initialize plot
        p = plot(title="Interpolation vs. Data for ρ=$(round(rho_val, digits=2))", 
                 xlabel="Assets (a)", ylabel=ylabel_text, legend=:bottomright)

        # Plot original data (discrete points)
        scatter!(p, a_grid, policy_matrix[i_rho, :], label="Raw Data", markersize=6, color=:blue)

        # Plot spline interpolation (smooth curve)
        plot!(p, a_dense, policy_spline_values, label="Spline Interpolation", linewidth=2, color=:red)

        # Display plot
        display(p)
    end
end

# plot_2d_policy_int_vs_data(policy_a, policy_a_int, a_grid, rho_grid, policy_type="assets")


################### PLOTTING OPTIMAL C AND L FROM LABOR FOC ###################

function plot_opt_c_l_from_FOC(a_i, opt_c_itp, opt_l_itp, a_grid, rho_grid)
    """
    Plots interpolated optimal consumption and labor as functions of a' for all levels of productivity ρ.

    Args:
        a           : Index of the chosen asset level (integer)
        opt_c_itp   : Dictionary of interpolated consumption functions {(ρ, a) => interp_c(a')}
        opt_l_itp   : Dictionary of interpolated labor functions {(ρ, a) => interp_l(a')}
        a_grid      : Grid of asset values
        rho_grid    : Grid of productivity levels
    """

    # Create plots for c*(a') and l*(a') as functions of a' for different ρ
    p1 = plot(title="Optimal Consumption c*(a') for a' = $(a_grid[a_i])", xlabel="a'", ylabel="c*", legend=:topright)
    p2 = plot(title="Optimal Labor l*(a') for a' = $(a_grid[a_i])", xlabel="a'", ylabel="l*", legend=:topleft)

    # Loop over productivity levels
    for rho_i in 1:length(rho_grid)
        # Extract the interpolation functions for the given (ρ, a)
        interp_c = opt_c_itp[rho_i, a_i]
        interp_l = opt_l_itp[rho_i, a_i]

        # Evaluate the interpolants over a' grid
        c_values = [interp_c(a_prime) for a_prime in a_grid]
        l_values = [interp_l(a_prime) for a_prime in a_grid]

        # Plot the interpolated policies
        plot!(p1, a_grid, c_values, label="ρ = $(rho_grid[rho_i])")
        plot!(p2, a_grid, l_values, label="ρ = $(rho_grid[rho_i])")
    end

    # Display the plots
    return p1, p2
end


#-#-#-#-#-#-#       2. PLOTTING VALUE AND POLICY FUNCTIONS        #-#-#-#-#-#-#

function plot_value_function(V_new, a_grid, rho_grid)
    """
    Plots the value function for each productivity level with interpolation.

    Args:
        V_new    : Matrix (N_rho × N_a) containing the value function.
        a_grid   : Vector of asset grid values.
        rho_grid : Vector of productivity grid values.

    Returns:
        Displays the interpolated value function plot.
    """

    # Initialize plot
    p = plot(title="Value Function by Productivity Level", 
             xlabel="Assets (a)", ylabel="Value Function V(a, ρ)", legend=:bottomright)

    # Loop over each productivity level
    for i_rho in 1:size(V_new, 1)
        # Interpolate value function using cubic splines
        itp = interpolate((a_grid,), V_new[i_rho, :], Gridded(Linear()))
        a_dense = range(minimum(a_grid), maximum(a_grid), length=200)  # Fine grid
        V_interp = [itp(a) for a in a_dense]

        # Plot interpolated curve
        plot!(p, a_dense, V_interp, label="ρ = $(rho_grid[i_rho])", lw=2)
    end

    # Display the plot
    return p
end

function plot_policy_function(policy_data, a_grid, rho_grid; policy_type="assets")
    """
    Generic function to plot policy functions (assets, labor, or consumption), detecting whether
    the input is a 2D matrix or a Spline2D interpolant.

    Args:
        policy_data : Either a (N_rho × N_a) matrix or a Spline2D interpolant.
        a_grid      : Vector of asset grid values.
        rho_grid    : Vector of productivity grid values.
        policy_type : String, either "assets", "labor", or "consumption" (determines labels).

    Returns:
        Displays the policy function plot.
    """

    # Define labels based on the selected policy function
    if policy_type == "assets"
        title_text = "Policy Function for Assets"
        ylabel_text = "Future Assets (a')"
        leg_pos = :bottomright
    elseif policy_type == "labor"
        title_text = "Policy Function for Labor"
        ylabel_text = "Labor Supplied (l)"
        leg_pos = :topright
    elseif policy_type == "consumption"
        title_text = "Policy Function for Consumption"
        ylabel_text = "Consumption (c)"
        leg_pos = :topleft
    else
        error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
    end

    # Initialize plot
    p = plot(title=title_text, xlabel="Current Assets (a)", ylabel=ylabel_text, legend=leg_pos)

    if isa(policy_data, Matrix)
        # Case: policy_data is a 2D matrix (raw data)
        for i_rho in 1:length(rho_grid)
            plot!(p, a_grid, policy_data[i_rho, :], label="ρ = $(rho_grid[i_rho])", lw=2)
        end
    elseif isa(policy_data, Spline2D)
        # Case: policy_data is a 2D spline interpolant
        a_dense = range(minimum(a_grid), maximum(a_grid), length=200)

        for rho_val in rho_grid
            policy_spline = [policy_data(rho_val, a) for a in a_dense]
            plot!(p, a_dense, policy_spline, label="ρ = $(round(rho_val, digits=4))", lw=2)
        end
    else
        error("Invalid input type for policy_data. Must be a 2D matrix or Spline2D.")
    end

    # Display the plot
    return p
end
