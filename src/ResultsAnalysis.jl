###############################################################################
############################# RESULTSANALYSIS.JL ##############################

######### This script analyses the results produced by simulations of #########
###################### the benchmark ProgTax(2025) model ######################

###############################################################################




#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------# 0. IMPORTING LIBRARIES AND SUBMODULES #------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


using LinearAlgebra
using Distances
using Base.Threads
using Interpolations
using DelimitedFiles
using DataFrames
using CairoMakie
using Plots
using Dates

include("AuxiliaryFunctions.jl")
include("PlottingFunctions.jl")
include("Parameters.jl")
include("AnalysisFunctions.jl")
include("Numerics.jl")
include("AnalysisFunctions.jl")

# Import also grids
@info("Making grids...")

# Define grid parameters
gpar = GridParams(a_min, 300.000, 400, # Assets
                    0.0, 1, 150,    # Labor
                    length(rho_grid) # Productivity 
                    )

# Assets
a_gtype = "polynomial"
a_grid = makeGrid(gpar.a_min, gpar.a_max, gpar.N_a; grid_type = a_gtype, pol_power = 4)

# Labor
l_grid = makeGrid(gpar.l_min, gpar.l_max, gpar.N_l; grid_type = "labor-double")




#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 1. IMPORTING MODEL RESULTS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Read results for baseline calibration
baseline_path = "output/baseline/model_results"
baseline = get_model_results(baseline_path)

# Clean baseline

# Get steady state values for multiple tax structs
folderpath = "output/equivalent_regimes"
# Retrieve model results folder
dirs = filter(isdir, readdir("output/equivalent_regimes", join=true))
dirs = joinpath.(dirs, "model_results")

# Extract data and append to baseline
ss = deepcopy(baseline)[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]

for i in 1:length(dirs)
    temp = get_model_results(dirs[i])
    append!(ss, temp)
end

# Reorder
sort!(ss, :tau_y)
# Exclude baseline temporarily - TBM
# ss = ss[setdiff(1:nrow(ss), [25]), :]

println(names(ss))

# plot_policy_function(ss.policy_c[1], a_grid, rho_grid; policy_type="consumption", 
#                      taxes=Taxes(baseline.taxes[1]...),
#                      cmap = :bluegreenyellow, # :Paired_7)
#                      reverse_palette = true)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 2. ADDING FURTHER INDICATORS #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Compute break-even points 
ss.bky = ss.lambda_y .^ (1 ./ ss.tau_y)
ss.bkc = ifelse.(ss.tau_c .> 0, ss.lambda_c .^ (1 ./ ss.tau_c), NaN)

#---------------------------------# 1. INCOME #--------------------------------#

# Compute Labor Income and Labor Income Tax Stats
# Preallocate new columns
ss.gini_income = similar(ss.lambda_y)
ss.policy_l_incpt = similar(ss.stat_dist)
ss.labor_tax_policy = similar(ss.stat_dist)
ss.labor_tax_rate = similar(ss.stat_dist)
ss.avg_income_stats = Vector{Any}(undef, nrow(ss))
ss.avg_rates_Wtax = Vector{Any}(undef, nrow(ss))
ss.t10tob50_inc_ratio = similar(ss.lambda_y)
ss.t10tob90_inc_ratio = similar(ss.lambda_y)
ss.aetr_Wtax = similar(ss.lambda_y)
ss.b50t10aetr_Wtax = Vector{Vector{Float64}}(undef, nrow(ss))

for i in 1:nrow(ss)
    distL = ss.distL[i]
    policy_l = ss.policy_l[i]
    w = ss.w[i]
    stat_dist = Matrix(ss.stat_dist[i])

    # Tax parameters from DataFrame columns
    lambda_y = ss.lambda_y[i]
    tau_y = ss.tau_y[i]

    # === Gini coefficient ===
    ss.gini_income[i] = compute_gini(distL .* w, stat_dist, plot_curve = false)

    # === Gross labor income ===
    policy_l_incpt = policy_l .* rho_grid .* w
    ss.policy_l_incpt[i] = policy_l_incpt

    labor_tax_policy = policy_l_incpt .- lambda_y .* policy_l_incpt .^ (1 - tau_y)
    labor_tax_rate = ifelse.((labor_tax_policy .== 0) .&& (policy_l_incpt .== 0),
                              0.0,
                              labor_tax_policy ./ policy_l_incpt)

    ss.labor_tax_policy[i] = labor_tax_policy
    ss.labor_tax_rate[i] = labor_tax_rate

    # === Income distribution stats ===
    ss.avg_income_stats[i] = compute_income_distribution_stats(stat_dist, policy_l_incpt)

    avg_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt; cutoffs = [0.5, 0.9, -0.1])
    ss.t10tob50_inc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[1][2]
    ss.t10tob90_inc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[2][2]

    # === Average effective rates and AETR ===
    ss.avg_rates_Wtax[i] = compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    ss.aetr_Wtax[i] = sum(labor_tax_rate .* stat_dist)
    ss.b50t10aetr_Wtax[i] = round.([ss.avg_rates_Wtax[i][2][2], ss.avg_rates_Wtax[i][4][2]], digits = 3)
end

#------------------------------# 2. CONSUMPTION #-----------------------------#
# Preallocate columns
ss.consumption_tax_policy = similar(ss.stat_dist)
ss.consumption_tax_rate = similar(ss.stat_dist)
ss.consumption_tax_gini = similar(ss.lambda_c)
ss.consumption_base_gini = similar(ss.lambda_c)
ss.kakwani_cons_tax = similar(ss.lambda_c)

for i in 1:nrow(ss)
    policy_c = ss.policy_c[i]
    stat_dist = Matrix(ss.stat_dist[i])

    lambda_c = ss.lambda_c[i]
    tau_c = ss.tau_c[i]

    # Compute consumption taxes paid
    consumption_tax_policy = policy_c .- lambda_c .* policy_c .^ (1 - tau_c)
    ss.consumption_tax_rate[i] = consumption_tax_policy ./ policy_c

    # Gini coefficients
    ss.consumption_tax_gini[i] = compute_gini(consumption_tax_policy, stat_dist; plot_curve = false)
    ss.consumption_base_gini[i] = compute_gini(policy_c .+ consumption_tax_policy, stat_dist; plot_curve = false)

    # Kakwani Index
    ss.kakwani_cons_tax[i] = ss.consumption_tax_gini[i] - ss.consumption_base_gini[i]

    # === Average effective rates and AETR ===
    ss.avg_rates_Ctax[i] = compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
end

# compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
#                            cutoffs = [0.1, 0.5, 0.9, -0.1])

# Get extrema for consumption tax rates
ss.cons_tax_rates_min = [round.(t; digits=3) for t in minimum.(ss.consumption_tax_rate)]
ss.cons_tax_rates_max = [round.(t; digits=3) for t in maximum.(ss.consumption_tax_rate)]

# Compute % variation from the baseline for each column
ss_pct = DataFrame()
float_vars = ["lambda_y", "tau_y", "lambda_c", "tau_c", "tau_k", 
              "aggC", "aggG", "aggH", "aggK", "aggL", "aggT_c", 
              "aggT_k", "aggT_y", "aggY", "r", "w"]

for col in float_vars
    baseline_val = baseline[1, col]
    ss_pct[!, col] = 100 .* (ss[!, col] .- baseline_val) ./ baseline_val
end

#---------------------------------# 3. TAXES #--------------------------------#

# Taxes - Revenue shares 
ss.TRevC = ss.aggT_c ./ ss.aggG
ss.TRevW = ss.aggT_y ./ ss.aggG
ss.TRevK = ss.aggT_k ./ ss.aggG

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------# 3. ANALYSING MAIN AGGREGATES #-----------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Choose extrema' indices for comparisons
idx_low_tau_y = 1
idx_high_tau_y = nrow(ss)
idx_middle = Int((nrow(ss) + 1) / 2)

# Generating dataframe for table collecting aggregate results 
agg_table = DataFrame(agg = String[], 
                      low = Float64[], 
                      mid = Float64[], 
                      high = Float64[])

#----# AGGREGATE TABLE SECTION 1 - TAXES #----#
# Add Government expenditure in first row
push!(agg_table, extract_low_mid_high(ss, :aggG; 
                                      table_string = "Government Expenditure",
                                      normalise = "num"))

# Add Revenue shares 
push!(agg_table, extract_low_mid_high(ss, :TRevC; 
                                      table_string = "Consumption Taxes - % of Revenue",
                                      as_percentage = true))
push!(agg_table, extract_low_mid_high(ss, :TRevW; 
                                      table_string = "Labor Income Taxes - % of Revenue",
                                      as_percentage = true))
push!(agg_table, extract_low_mid_high(ss, :TRevK; 
                                      table_string = "Capital Return Taxes - % of Revenue",
                                      as_percentage = true))


# Plot and add main aggregates to DataFrame for table
# Output
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggY;
                     ylabel = "Aggregate Output",
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :aggY; table_string = "Agg. Output"))

# Consumption                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggC;
                     ylabel = "Aggregate Consumption", 
                     cmap = :inferno)

push!(agg_table, extract_low_mid_high(ss, :aggC; table_string = "Agg. Consumption"))

# Capital                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggK;
                     ylabel = "Aggregate Capital", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :aggK; table_string = "Agg. Capital"))

# Hours worked                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggH;
                     ylabel = "Aggregate Hours Worked", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :aggH; table_string = "Agg. Hours Worked"))

# Prices
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.r;
                     ylabel = "Interest Rate", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :r; table_string = "Interest Rate"))
            
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.w;
                     ylabel = "Equilibrium Wage", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :w; table_string = "Wage"))

#-----# Inequalities #-----#

# Income
# Gini for pre-tax income
plot_colored_scatter(ss.tau_y, ss.tau_c, ss.gini_income;
                     ylabel = "Gini Coefficient for Pre-Tax Income", 
                     cmap = :avocado)

# Top 10% to Bottom 90% Income Share Ratio
plot_colored_scatter(ss.tau_y, ss.tau_c, ss.t10tob90_inc_ratio;
                     ylabel = "Top 10% - Bottom 90% Income Share Ratio", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :t10tob90_inc_ratio; table_string = "T10B90 Income Ratio"))


#-----# Taxes #-----#
# Consumption
# Kakwani Consumption Tax
plot_colored_scatter(ss.tau_y, ss.tau_c, ss.kakwani_cons_tax;
                     ylabel = "Kakwani Index for Consumption Tax", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :kakwani_cons_tax; table_string = "Kakwani Index for Consumption Tax"))


# Min and max Consumption Tax Rates
plot_colored_scatter(ss.tau_y, ss.tau_c, ss.cons_tax_rates_max .- ss.cons_tax_rates_min;
                     ylabel = "Consumption Tax Rate Range Width", 
                     cmap = :avocado)

push!(agg_table, extract_low_mid_high(ss, :cons_tax_rates_min; table_string = "Kakwani Index for Consumption Tax"))
t1 = extract_low_mid_high(ss, :cons_tax_rates_min; table_string = "cons_tax_rates_min")
t2 = extract_low_mid_high(ss, :cons_tax_rates_max; table_string = "cons_tax_rates_max")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 4. ANALYSING DISTRIBUTIONS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Analyse distributional effects - who gained, who lost?
# Consumption #
# Plot
densC = build_density_dict(ss, :distC, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densC, [:low, :mid, :high];
                        ylabel = "Consumption (c)", title = "Consumption Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Check and plot differences btw high progressivity and low progressivity regime
densC_diff = densC[:high] - densC[:low]


# Capital
densK = build_density_dict(ss, :distK, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densK, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Wealth Held (density)", title = "Asset distribution",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Hours worked
densH = build_density_dict(ss, :distH, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densH, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Wealth Held (density)", title = "Asset distribution",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Labor Taxes
densWtax = build_density_dict(ss, :distWtax, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densWtax, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Labor Income Taxes", title = "Labor Income Tax Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (1, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Consumption taxes
densCtax = build_density_dict(ss, :distCtax, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densCtax, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Consumption Taxes", title = "Consumption Tax Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (1, 240),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

















################################################################################


# # Plot consumption by progressivity 
# plot_aggregate_surface(ss.aggC, ss.tau_c, ss.tau_y;
#                         zlabel = "Consumption",
#                         title_text = "Aggregate consumption by tax progressivity",
#                         cmap = :avocado,
#                         azimuth = 5π/4)

# # Government expenditure
# plot_aggregate_surface(ss.G, ss.tau_c, ss.tau_y;
#                     zlabel = "Government expenditure",
#                     title_text = "Government expenditure by tax progressivity",
#                     cmap = :linear_bgyw_20_98_c66_n256)

# # Savings/Capital
# plot_aggregate_surface(ss.aggK, ss.tau_c, ss.tau_y;
#                     zlabel = "Aggregate savings",
#                     title_text = "Aggregate savings by tax progressivity",
#                     cmap = :summer)

# # Interest rate                    
# plot_aggregate_surface(ss.r, ss.tau_c, ss.tau_y;
#                     zlabel = "Interest rate",
#                     title_text = "Equilibrium rate by tax progressivity",
#                     cmap = :heat,
#                     azimuth = 5π/4)

# # Wage
# plot_aggregate_surface(ss.w, ss.tau_c, ss.tau_y;
#                     zlabel = "Wage",
#                     title_text = "Equilibrium wage by tax progressivity",
#                     cmap = :haline)

# # Interest rate
