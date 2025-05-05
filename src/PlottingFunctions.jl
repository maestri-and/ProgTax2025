###############################################################################
############################ PLOTTINGFUNCTIONS.JL #############################

###### This script defines useful plotting functions for visual insights ######
################### from the benchmark ProgTax(2025) model ####################

###############################################################################


using StatsBase
using Plots
using CairoMakie # Switch to GLMakie for interactive plots
using LaTeXStrings



###############################################################################
####################### 0. PRELIMINARY PLOTS - UTILITY ########################
###############################################################################

# Plotting generic 1D f(x) vs x
function plot_f(f; x_min=-1, x_max=1)
    cs = range(x_min, x_max, length=500)
    fs = [f(c) for c in cs]
    Plots.plot(cs, fs, xlabel="c", ylabel="f(c)", title="Objective function", legend=false)
    hline!([0], linestyle=:dash, color=:red) # Show where f(c) = 0
end

# plot_f(f, x_min = -1, x_max = 5)

# Plotting dummy 3D utility function 

# Function to plot the utility function and return the utility matrix and maximum point
function plot_utility_function(rra, phi, frisch; normalise = false, c_range = (0.1, 5.0), l_range = (0.1, 5.0), num_points = 100)
    # Generate a range of consumption and labor values
    c_values = range(c_range..., length = num_points)
    l_values = range(l_range..., length = num_points)

    # Create a grid of consumption and labor values
    utility_matrix = [get_utility_hh(c, l, hhpar, normalise = normalise) for c in c_values, l in l_values]

    # Transpose the utility matrix for correct plotting
    utility_matrix = utility_matrix'

    # Plot the utility function
    p = Plots.plot(c_values, l_values, utility_matrix, st = :surface, xlabel = "Consumption (c)", ylabel = "Labor (l)", zlabel = "Utility", title = "Utility Function")

    # Find the maximum utility value and its coordinates
    max_utility = maximum(utility_matrix)
    max_index = argmax(utility_matrix)
    max_c = c_values[max_index[1]]
    max_l = l_values[max_index[2]]

    @info("Maximum utility: $max_utility at (c = $max_c, l = $max_l)")

    # Return utility matrix and plot
    return utility_matrix, p, (max_utility, max_c, max_l)
end

# # Example usage
# ut_matrix, utility_plot, max_point = plot_utility_function(2.0, 1.0, 0.5; normalise = false)

# utility_plot
# max_point

# Function to plot the utility function and return the utility matrix and maximum point
# Imposing budget constraint to hold for given a, rho, a'

function plot_utility_with_bc(rra, phi, frisch; a_i = 10, a_prime_i = 10, rho_i = 3, normalise = false, l_grid = l_grid)
    # Choose labor grid
    l_values = l_grid

    # Compute household taxes, consumption, and utility
    @elapsed _, hh_consumption, _, hh_utility = compute_hh_taxes_consumption_utility_(a_grid,
                                                                    gpar.N_a, rho_grid, l_values, w, r, taxes, hhpar)

    # Fix one level of a and a'
    c_values = hh_consumption[:, rho_i, a_i, a_prime_i]
    utility_values = hh_utility[:, rho_i, a_i, a_prime_i]

    # Plot consumption and labor
    p1 = Plots.plot(l_values, c_values, xlabel = "Labor (l)", ylabel = "Consumption (c)", title = "Consumption and Utility vs. Labor - Fixed ρ, a and a'", label = "Consumption")

    # Plot utility and labor
    p2 = Plots.plot(l_values, utility_values, xlabel = "Labor (l)", ylabel = "Utility", label = "Utility", linecolor = :red)

    # Combine plots
    p = Plots.plot(p1, p2, layout = (2, 1), size = (800, 600))

    # Find the maximum utility value and its coordinates
    max_utility = maximum(utility_values)
    max_index = argmax(utility_values)
    max_c = c_values[max_index]
    max_l = l_values[max_index]

    @info("Maximum utility: $max_utility at (c = $max_c, l = $max_l)")

    # Return utility values and plot
    return utility_values, p, (max_utility, max_c, max_l)
end

# utility_values, p, (max_utility, max_c, max_l) = plot_utility_with_bc(2.0, 1.0, 0.5; a_i = 10, a_prime_i = 10, rho_i = 3, normalise = false, l_grid = l_grid)
# p #Display graphs
# savefig(p, "output/preliminary/utility_consumption_labor_budget_constraint.png")


###############################################################################
############################## 1. PLOTTING TAXES ##############################
###############################################################################


# Plot for single rate of progressivity
# Flatten matrices into column vectors
# Generic function to plot taxes vs tax base
function plot_taxes_vs_base(base_vec, tax_vec; base_range=nothing)
    if isnothing(base_range)
        # Sample a subset of data points for plotting
        sample_indices = sample(1:length(base_vec), min(10000, length(base_vec)), replace=false)

        # Plot the full dataset
        p = Plots.scatter(base_vec[sample_indices], tax_vec[sample_indices],
                    xlabel="Tax Base", ylabel="Tax",
                    title="Taxes vs Tax Base", legend=false, markercolor=:green)
    else
        # Filter the data based on the specified base range
        lower_bound, upper_bound = base_range
        filter_indices = findall(x -> lower_bound < x < upper_bound, base_vec)

        # Plot the filtered dataset
        p = Plots.scatter(base_vec[filter_indices], tax_vec[filter_indices],
                    xlabel="Tax Base", ylabel="Tax",
                    title="Taxes vs Tax Base \n Range-Adjusted", legend=false, markercolor=:red)
    end

    # Add a horizontal line at tax = 0 for reference
    hline!(p, [0], linestyle=:dash, color=:black, lw=2, label="Tax = 0")

    # Return the plot
    return p
end

# plot_taxes_vs_base(vec(hh_consumption), vec(hh_consumption_tax); base_range = (-0.1, 0.5))

# function plot_1D_itp_vs_data(itp, x_data, y_data; x_range = nothing, y_range = nothing)
#     if x_range

###############################################################################
######################### 2. INTERPOLATIONS VS DATA ###########################
###############################################################################

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
    p = Plots.plot(x_subset[idx], y_subset[idx],
         seriestype=:scatter, label="Data", legend=:topleft, framestyle=:box,
         xlabel="x", ylabel="y", title="Interpolation vs Data")

    # Plot the interpolation
    Plots.plot!(p, xq, itp.(xq), label="Interpolation", linewidth=2)

    # Optionally adjust y-axis range
    if y_range !== nothing
        ylims!(y_range)
    end

    return p
end

# plot_1D_itp_vs_data(itp, x_data, y_data; x_range = (-0.0001, 0.0005))
# plot_1D_itp_vs_data(opt_c_itp[7, 50], a_grid, opt_c_FOC[7,50,:]; x_range = (-1, 5))


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
        @error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
        error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
    end

    # Loop over each productivity level ρ and create a plot
    for i_rho in 1:length(rho_grid)
        rho_val = rho_grid[i_rho]

        # Create a denser asset grid for interpolation
        a_dense = range(minimum(a_grid), maximum(a_grid), length=200)
        policy_spline_values = [policy_spline(rho_val, a) for a in a_dense]

        # Initialize plot
        p = Plots.plot(title="Interpolation vs. Data for ρ=$(round(rho_val, digits=2))", 
                 xlabel="Assets (a)", ylabel=ylabel_text, legend=:bottomright)

        # Plot original data (discrete points)
        Plots.scatter!(p, a_grid, policy_matrix[i_rho, :], label="Raw Data", markersize=6, color=:blue)

        # Plot spline interpolation (smooth curve)
        Plots.plot!(p, a_dense, policy_spline_values, label="Spline Interpolation", linewidth=2, color=:red)

        # Display plot
        display(p)
    end
end

# plot_2d_policy_int_vs_data(policy_a, policy_a_int, a_grid, rho_grid, policy_type="assets")


###############################################################################
################# 3. PLOTTING OPTIMAL C AND L FROM LABOR FOC ##################
###############################################################################


function plot_utility_implied_by_labor_FOC(rho, w, taxes, hhpar)
    # Define the consumption grid
    c_grid = range(-5.0, 5.0, length=50)  # Reduce size for performance
    l_grid = range(0.0, 1.0, length=50)  # Define labor grid

    # Create a meshgrid for c and l
    C = repeat(c_grid, 1, length(l_grid))  # 2D matrix for c values
    L = repeat(l_grid', length(c_grid), 1)  # 2D matrix for l values

    # Compute utility for each (c, l) pair
    U = [get_utility_hh(c, l, hhpar) for (c, l) in zip(C, L)]

    # Create interactive 3D plot
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1, 1], xlabel="Consumption (c)", ylabel="Labor (l)", zlabel="Utility U(c,l)", title="Utility Implied by Labor FOC")

    CairoMakie.surface!(ax, C, L, U, colormap=:viridis)

    display(fig)  # Show the figure
end

# plot_utility_implied_by_labor_FOC(rho_grid[1], 1, taxes, hhpar)


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
    p1 = Plots.plot(title="Optimal Consumption c*(a') for a = $(a_grid[a_i])", xlabel="a'", ylabel="c*", legend=:topright)
    p2 = Plots.plot(title="Optimal Labor l*(a') for a = $(a_grid[a_i])", xlabel="a'", ylabel="l*", legend=:topleft)

    # Loop over productivity levels
    for rho_i in 1:length(rho_grid)
        # Extract the interpolation functions for the given (ρ, a)
        interp_c = opt_c_itp[rho_i, a_i]
        interp_l = opt_l_itp[rho_i, a_i]

        # Evaluate the interpolants over a' grid
        c_values = [interp_c(a_prime) for a_prime in a_grid]
        l_values = [interp_l(a_prime) for a_prime in a_grid]

        # Plot the interpolated policies
        Plots.plot!(p1, a_grid, c_values, label="ρ = $(rho_grid[rho_i])")
        Plots.plot!(p2, a_grid, l_values, label="ρ = $(rho_grid[rho_i])")
    end

    # Display the plots
    return p1, p2
end

# p1, p2 = plot_opt_c_l_from_FOC(1, opt_c_itp, opt_l_itp, a_grid, rho_grid)
# display(p2)


###############################################################################
################### 4. PLOTTING VALUE AND POLICY FUNCTIONS ####################
###############################################################################

function plot_value_function(V_new, a_grid, rho_grid; taxes=taxes)
    """
    Plots the value function for each productivity level with interpolation.

    Args:
        V_new    : Matrix (N_rho × N_a) containing the value function.
        a_grid   : Vector of asset grid values.
        rho_grid : Vector of productivity grid values.

    Returns:
        Displays the interpolated value function plot.
    """

    # Subtitle string with tax parameters
    tax_info = "λ_y=$(taxes.lambda_y), τ_y=$(taxes.tau_y), λ_c=$(taxes.lambda_c), τ_c=$(taxes.tau_c), τ_k=$(taxes.tau_k)"

    # Define main title and subtitle
    main_title = "Value Function by Productivity Level"
    tax_subtitle = LaTeXString("\$$(tax_info)\$")

    # Title-only dummy plot (acts as suptitle)
    suptitle = Plots.plot(title=main_title, grid=false, showaxis=false, bottom_margin=-35Plots.px)

    # Actual content plot
    p = Plots.plot(title=tax_subtitle, xlabel="Assets (a)", ylabel="Value Function V(a, ρ)", legend=:bottomright)

    for i_rho in 1:size(V_new, 1)
        itp = interpolate((a_grid,), V_new[i_rho, :], Gridded(Linear()))
        a_dense = range(minimum(a_grid), maximum(a_grid), length=200)
        V_interp = [itp(a) for a in a_dense]
        Plots.plot!(p, a_dense, V_interp, label="ρ = $(rho_grid[i_rho])", lw=2)
    end

    return Plots.plot(suptitle, p, layout = @layout([A{0.01h}; B]))
end

function plot_policy_function(policy_data, a_grid, rho_grid; policy_type="assets", taxes=taxes)
    """
    Generic function to plot policy functions (assets, labor, or consumption), detecting whether
    the input is a 2D matrix, Spline2D, GriddedInterpolation, or a generic (ρ, a) -> value function.

    Args:
        policy_data : Matrix, Spline2D, GriddedInterpolation, or function.
        a_grid      : Vector of asset grid values.
        rho_grid    : Vector of productivity grid values.
        policy_type : String, either "assets", "labor", or "consumption".

    Returns:
        A plot of the policy function.
    """

    # Define plot labels
    titles = Dict(
        "assets" => ("Policy Function for Assets", "Future Assets (a')", :bottomright),
        "labor" => ("Policy Function for Labor", "Labor Supplied (l)", :topright),
        "consumption" => ("Policy Function for Consumption", "Consumption (c)", :topleft)
    )

    haskey(titles, policy_type) || @error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
    title_text, ylabel_text, leg_pos = titles[policy_type]
    
    # Subtitle string with tax parameters
    tax_info = "λ_y=$(taxes.lambda_y), τ_y=$(taxes.tau_y), λ_c=$(taxes.lambda_c), τ_c=$(taxes.tau_c), τ_k=$(taxes.tau_k)"

    # Initialize plot
    policy_title = title_text 
    tax_subtitle = LaTeXString("\$$(tax_info)\$")

    suptitle = Plots.plot(title = policy_title, grid = false, showaxis = false, bottom_margin = -35Plots.px)
    
    p = Plots.plot(title = tax_subtitle, xlabel="Current Assets (a)", ylabel=ylabel_text, legend=leg_pos)

    a_dense = range(minimum(a_grid), maximum(a_grid), length=200)

    if typeof(policy_data) <: Function || typeof(policy_data) <: Interpolations.GriddedInterpolation
        # Callable interpolants: Spline2D, clamped func, GriddedInterpolation
        for rho_val in rho_grid
            vals = [policy_data(rho_val, a) for a in a_dense]
            Plots.plot!(p, a_dense, vals, label="ρ = $(round(rho_val, digits=4))", 
                     lw=2, title = tax_subtitle)
        end
    
    elseif isa(policy_data, AbstractMatrix)
        # Raw matrix input
        for i_rho in 1:length(rho_grid)
            Plots.plot!(p, a_grid, policy_data[i_rho, :], label="ρ = $(rho_grid[i_rho])", 
                     lw=2, title = tax_subtitle)
        end

    else
        @error("Invalid input type for policy_data.")
        error("Invalid input type for policy_data.")
    end

    return Plots.plot(suptitle, p, layout = @layout([A{0.01h}; B]))
end

function plot_household_policies(valuef, policy_a_int, policy_l_int, policy_c,
                                 a_grid, rho_grid, taxes;
                                 plot_types = ["value", "assets", "labor", "consumption"],
                                 save_plots = false)

    """
    Plots household value and policy functions.

    Args:
        valuef         : Value function matrix (ρ × a)
        policy_a_int   : Interpolated asset policy (Spline2D)
        policy_l_int   : Interpolated labor policy (Spline2D)
        policy_c       : Consumption policy matrix (ρ × a)
        a_grid         : Asset grid
        rho_grid       : Productivity grid
        taxes          : Struct containing tax parameters
        plot_types     : Vector of strings selecting which plots to show
        save_plots     : If true, saves plots to predefined file paths
    """

    if "value" in plot_types
        p_val = plot_value_function(valuef, a_grid, rho_grid)
        display(p_val)
        if save_plots
            savefig(p_val, "output/preliminary/policy_funs/cont/value_function_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")
        end
    end

    if "assets" in plot_types
        p_a = plot_policy_function(policy_a_int, a_grid, rho_grid, policy_type = "assets")
        display(p_a)
        if save_plots
            savefig(p_a, "output/preliminary/policy_funs/cont/asset_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")
        end
    end

    if "labor" in plot_types
        p_l = plot_policy_function(policy_l_int, a_grid, rho_grid, policy_type = "labor")
        display(p_l)
        if save_plots
            savefig(p_l, "output/preliminary/policy_funs/cont/labor_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")
        end
    end

    if "consumption" in plot_types
        p_c = plot_policy_function(policy_c, a_grid, rho_grid, policy_type = "consumption")
        display(p_c)
        if save_plots
            savefig(p_c, "output/preliminary/policy_funs/cont/cons_policy_ly$(taxes.lambda_y)_ty$(taxes.tau_y)_lc$(taxes.lambda_c)_tc$(taxes.tau_c).png")
        end
    end
end

# 3D plot

function plot_policy_function_3d(policy_data, a_grid, rho_grid; policy_type="labor", taxes=taxes)
    titles = Dict(
        "assets"      => ("Asset Policy Function", "Future Assets"),
        "labor"       => ("Labor Policy Function", "Labor Supply"),
        "consumption" => ("Consumption Policy Function", "Consumption")
    )

    haskey(titles, policy_type) || error("Invalid policy_type. Choose 'assets', 'labor', or 'consumption'.")
    title_text, zlabel_text = titles[policy_type]

    a_dense = range(minimum(a_grid), maximum(a_grid), length=100)
    ρ_dense = range(minimum(rho_grid), maximum(rho_grid), length=length(rho_grid))

    A = repeat(a_dense', length(ρ_dense), 1)
    R = repeat(ρ_dense, 1, length(a_dense))
    Z = similar(A)

    if typeof(policy_data) <: Function || typeof(policy_data) <: Interpolations.GriddedInterpolation
        for i in 1:length(ρ_dense), j in 1:length(a_dense)
            Z[i, j] = policy_data(ρ_dense[i], a_dense[j])
        end
    elseif isa(policy_data, AbstractMatrix)
        interp = Interpolations.interpolate((rho_grid, a_grid), policy_data, Interpolations.Gridded(Interpolations.Linear()))
        for i in 1:length(ρ_dense), j in 1:length(a_dense)
            Z[i, j] = interp(ρ_dense[i], a_dense[j])
        end
    else
        error("Invalid input type for policy_data.")
    end

    fig = CairoMakie.Figure()

    # Title
    CairoMakie.Label(
    fig[1, 1, Top()], title_text;
    fontsize = 18,
    halign = :center,
    padding = (0, 0, 10, 0))

    # Adjust distance
    # CairoMakie.rowgap!(fig.layout, -250)

    ax = CairoMakie.Axis3(
    fig[1, 1],
    # title = title_text,
    xlabel = "Asset holdings",
    ylabel = "Skill level",
    zlabel = zlabel_text,
    azimuth = -.25π,    # horizontal rotation (left-right)
    # elevation = 30    # vertical tilt
    )   
    # my_cmap = CairoMakie.cgrad(
    #     [:deepskyblue, :deepskyblue, :lime, :yellow, :yellow],  # more blue/yellow anchors
    #     [0.0, 0.2, 0.5, 0.8, 1.0],                # positions in [0,1]
    #     categorical = false
    # )

    CairoMakie.surface!(ax, A, R, Z, 
                        colormap = :turbo,
                        shading = NoShading,
                        transparency = false,
                        colorrange = extrema(Z))

    CairoMakie.wireframe!(ax, A, R, Z; overdraw = true, color = :black, linewidth = 0.3)

    
    return(fig)
end

###############################################################################
###############################################################################
##################### 5. PLOTTING AGGREGATE DISTRIBUTIONS #####################
###############################################################################
###############################################################################

function plot_heatmap_stationary_distribution(stat_dist; taxes=taxes)
    # Initialise title
    title_text = "Stationary distribution of households across states"
    tax_info = "λ_y=$(taxes.lambda_y), τ_y=$(taxes.tau_y), λ_c=$(taxes.lambda_c), τ_c=$(taxes.tau_c), τ_k=$(taxes.tau_k)"

    # Initialize plot
    p = Plots.heatmap(stat_dist,
    title=LaTeXString("$title_text\n\$$(tax_info)\$\n"), 
    xlabel="Wealth level (a)", ylabel="Productivity level (ρ)",
    color = :heat,                # or :heat
    background_color = :white,
    colorbar_title = "\nDensity",
    colorbar_titlefont = 10,
    right_margin = 5Plots.mm,
    framestyle = :box)

    return(p)
end

function plot_density_by_productivity(stat_dist, a_grid, gpar; rho_grid=nothing)

    plt = Plots.plot(
        xlabel = "Wealth (a)",
        ylabel = "Density",
        title = "Stationary distribution by productivity level",
        legend = :topright,
        linewidth = 2
    )

    for i in 1:gpar.N_rho
        label = isnothing(rho_grid) ? "ρ = $i" : "ρ = $(round(rho_grid[i], digits=2))"
        Plots.plot!(plt, a_grid, stat_dist[i, :], label = label)
    end

    return plt
end

function plot_wealth_dist_bar(stats::Vector; title_str::String = "Wealth Distribution", show_labels::Bool = true)
    """
    plot_wealth_dist_bar(stats::Vector{Tuple}; title_str="Wealth Distribution", show_labels=true)

    Plots a bar chart from a vector of tuples of the form:
      (quantile_ref, share, _) where:
      - `quantile_ref` is a Float or Tuple (e.g., 0.5, -0.1, (0.3, 0.7))
      - `share` is the share of wealth held (e.g., 0.25 for 25%)
      - the third element is ignored

    Negative values in `quantile_ref` are treated as "Top x%" (e.g., -0.1 → Top 10%)
    Positive values are treated as "Bottom x%" or "Middle x–y%" ranges.

    Options:
      - `show_labels=true`: display rounded percentage labels on top of bars
    """

    labels = String[]
    shares = Float64[]

    # Parse input
    for (ref, share, _) in stats
        label = ref isa Tuple ? 
            "Middle $(Int(ref[1]*100))-$(Int(ref[2]*100))%" :
            (ref < 0 ? "Top $(Int(-ref*100))%" : "Bottom $(Int(ref*100))%")
        push!(labels, label)
        push!(shares, share * 100)
    end

    fig = CairoMakie.Figure(size = (800, 400))
    ax = CairoMakie.Axis(fig[1, 1]; title=title_str, xlabel="Group", ylabel="Wealth Share (%)",
                         xticks=(1:length(labels), labels))

    # Plot bars
    CairoMakie.barplot!(ax, 1:length(shares), shares; color=:limegreen, width=0.6)

    # Add labels
    if show_labels
        for (i, y) in enumerate(shares)
            CairoMakie.text!(ax, i, y + 2, text="$(round(y, digits=1))%", align=(:center, :bottom), fontsize=14)
        end
    end

    return fig
end

function plot_model_vs_data(data::Vector{<:Real}, model::Vector{<:Real}, labels::Vector{<:AbstractString};
    title_str::String = "Model vs Data", ylabel_str::String = "Percentage",
    barcolor = :limegreen)
@assert length(data) == length(model) == length(labels) "Vectors must be the same length"

fig = CairoMakie.Figure(size = (800, 400))
ax = CairoMakie.Axis(fig[1, 1], title=title_str, xlabel="Group", ylabel=ylabel_str,
 xticks=(1:length(labels), labels))

# Bar plot for data
CairoMakie.barplot!(ax, 1:length(data), data; color=barcolor, width=0.6, transparency=true)

# Dot markers for model results
CairoMakie.scatter!(ax, 1:length(model), model; color=:black, markersize=10)

# Legend
CairoMakie.axislegend(ax,
    [CairoMakie.PolyElement(color=barcolor), CairoMakie.MarkerElement(marker=:circle, color=:black)],
    ["Data", "Model"]
)

return fig
end

###############################################################################
############# 6. PLOTTING AGGREGATES BY PROGRESSIVITY PARAMETERS ##############
###############################################################################


function plot_aggregate_surface(aggtoplot_vec, tau_c_vec, tau_y_vec;
    xlabel="Consumption Tax Progressivity", 
    ylabel="Labor Tax Progressivity",
    # xlabel=LaTeXString("Consumption Tax Progressivity \$(\\tau_c)\$"), 
    # ylabel=LaTeXString("Labor Tax Progressivity \$(\\tau_y)\$"), 
    zlabel="Aggregate", title_text="3D Surface Plot",
    cmap = :haline, wireframe = true,
    azimuth = π/4, elevation = π/6)

    # Get unique sorted values for the grid axes
    tau_c_grid = sort(unique(tau_c_vec))
    tau_y_grid = sort(unique(tau_y_vec))

    # Create the grid (meshgrid)
    C = repeat(tau_c_grid', length(tau_y_grid), 1)  # tau_c on x-axis
    Y = repeat(tau_y_grid, 1, length(tau_c_grid))   # tau_y on y-axis

    # Fill Z surface: reshape assuming data is ordered row-wise by tau_y, tau_c
    Z = reshape(aggtoplot_vec, length(tau_y_grid), length(tau_c_grid))

    # Plotting
    fig = Figure(size=(800, 600))

    # Title
    CairoMakie.Label(
        fig[1, 1, Top()], title_text;
        fontsize = 20,
        halign = :center,
        padding = (0, 0, 10, 0))

    ax = Axis3(fig[1, 1], xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
               xlabelsize = 16, ylabelsize = 16, zlabelsize = 16,
               xticklabelsize = 12, yticklabelsize = 12, zticklabelsize = 12,
               azimuth = azimuth,    # horizontal rotation (left-right)
               elevation = elevation # vertical tilt
               )

    CairoMakie.surface!(ax, C, Y, Z, 
                        colormap=cmap, 
                        shading = NoShading,
                        colorrange=extrema(Z))
    
    if wireframe                    
        CairoMakie.wireframe!(ax, C, Y, Z; color=:black, transparency=true)
    end

    return(fig)
end    