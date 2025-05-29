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
using PrettyTables  

include("AuxiliaryFunctions.jl")
include("PlottingFunctions.jl")
include("Parameters.jl")
include("AnalysisFunctions.jl")
include("Numerics.jl")

# Define if needing to keep baseline in dataset
keep_baseline = true

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 1. IMPORTING MODEL RESULTS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Import also grids
@info("Importing Model Results...")

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

# Read results for baseline calibration
baseline_path = "output/equilibria/baseline/model_results"
baseline = get_model_results(baseline_path)

# Ensure data type
matrix_cols = ["distC", "distCtax", "distH", "distK", "distKtax", "distL", "distWtax",
               "policy_a", "policy_c", "policy_l", "stat_dist"]
for col in matrix_cols
    baseline[!, col] = Matrix.(baseline[!, col])
end

baseline.taxes = Taxes.(baseline.taxes)

# Get steady state values for multiple tax structs
folderpath = "output/equilibria/equivalent_regimes"
# Retrieve model results folder
dirs = filter(isdir, readdir("output/equilibria/equivalent_regimes", join=true))
dirs = joinpath.(dirs, "model_results")

# Extract data and append to baseline
ss = deepcopy(baseline)[:, Not([:fpar, :gpar, :hhpar, :rhopar, :taxes])]

for i in 1:length(dirs)
    temp = get_model_results(dirs[i])
    append!(ss, temp)
end

# Remove baseline if needed 
if !keep_baseline
    ss = ss[2:end, :]
end

# Reorder
sort!(ss, :tau_y)

println(names(ss))


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
ss.gini_disp_income = similar(ss.lambda_y)
ss.policy_l_incpt = similar(ss.stat_dist)
ss.labor_tax_policy = similar(ss.stat_dist)
ss.labor_tax_rate = similar(ss.stat_dist)
ss.income_dist_stats = Vector{Any}(undef, nrow(ss))
ss.avg_rates_Wtax = Vector{Any}(undef, nrow(ss))
ss.t10tob50_ptinc_ratio = similar(ss.lambda_y)
ss.t10tob90_ptinc_ratio = similar(ss.lambda_y)
ss.t10tob50_atinc_ratio = similar(ss.lambda_y)
ss.t10tob90_atinc_ratio = similar(ss.lambda_y)
ss.aetr_Wtax = similar(ss.lambda_y)
ss.b50t10aetr_Wtax = Vector{Any}(undef, nrow(ss))

for i in 1:nrow(ss)
    distL = ss.distL[i]
    policy_l = ss.policy_l[i]
    w = ss.w[i]
    stat_dist = Matrix(ss.stat_dist[i])

    # Tax parameters from DataFrame columns
    lambda_y = ss.lambda_y[i]
    tau_y = ss.tau_y[i]

    # === Gross labor income ===
    policy_l_incpt = policy_l .* rho_grid .* w
    ss.policy_l_incpt[i] = policy_l_incpt

    labor_tax_policy = policy_l_incpt .- lambda_y .* policy_l_incpt .^ (1 - tau_y)
    labor_tax_rate = ifelse.((labor_tax_policy .== 0) .&& (policy_l_incpt .== 0),
                              0.0,
                              labor_tax_policy ./ policy_l_incpt)

    ss.labor_tax_policy[i] = labor_tax_policy
    ss.labor_tax_rate[i] = labor_tax_rate

     # === Gini coefficient ===
    ss.gini_income[i] = compute_gini(policy_l_incpt, stat_dist, plot_curve = false)
    ss.gini_disp_income[i] = compute_gini(policy_l_incpt .- labor_tax_policy, stat_dist, plot_curve = false)

    # === Income distribution stats ===
    ss.income_dist_stats[i] = compute_income_distribution_stats(stat_dist, policy_l_incpt)

    avg_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt; cutoffs = [0.5, 0.9, -0.1])
    ss.t10tob50_ptinc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[1][2]
    ss.t10tob90_ptinc_ratio[i] = avg_income_stats[3][2] / avg_income_stats[2][2]

    avg_disp_income_stats = compute_average_income_stats(stat_dist, policy_l_incpt .- labor_tax_policy; cutoffs = [0.5, 0.9, -0.1])
    ss.t10tob50_atinc_ratio[i] = avg_disp_income_stats[3][2] / avg_disp_income_stats[1][2]
    ss.t10tob90_atinc_ratio[i] = avg_disp_income_stats[3][2] / avg_disp_income_stats[2][2]

    # === Average effective rates and AETR ===
    ss.avg_rates_Wtax[i] = compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    ss.aetr_Wtax[i] = sum(labor_tax_rate .* stat_dist)
    ss.b50t10aetr_Wtax[i] = round.((ss.avg_rates_Wtax[i][2][2], ss.avg_rates_Wtax[i][4][2]), digits = 3)
end

#------------------------------# 2. CONSUMPTION #-----------------------------#
# Preallocate columns
ss.consumption_tax_policy = similar(ss.stat_dist)
ss.consumption_tax_rate = similar(ss.stat_dist)
ss.consumption_tax_rate_vat = similar(ss.stat_dist)
ss.consumption_tax_gini = similar(ss.lambda_c)
ss.consumption_base_gini = similar(ss.lambda_c)
ss.consumption_gini = similar(ss.lambda_c)
ss.kakwani_cons_tax = similar(ss.lambda_c)
ss.cons_dist_stats = Vector{Any}(undef, nrow(ss))
ss.t10tob50_cons_ratio = similar(ss.lambda_c)
ss.t10tob90_cons_ratio = similar(ss.lambda_c)
ss.avg_rates_Ctax = Vector{Any}(undef, nrow(ss))
ss.avg_rates_Ctax_vat = Vector{Any}(undef, nrow(ss))
ss.b10t10_rates_Ctax = Vector{Any}(undef, nrow(ss))
ss.cons_deciles = Vector{Vector{Float64}}(undef, nrow(ss))

for i in 1:nrow(ss)
    policy_c = ss.policy_c[i]
    stat_dist = Matrix(ss.stat_dist[i])

    lambda_c = ss.lambda_c[i]
    tau_c = ss.tau_c[i]

    # Compute consumption distribution deciles
    ss.cons_deciles[i] = compute_decile_distribution(ss.stat_dist[i], ss.policy_c[i])

    # Compute consumption taxes paid
    consumption_tax_policy = policy_c .- lambda_c .* policy_c .^ (1 - tau_c)
    ss.consumption_tax_policy[i] = consumption_tax_policy
    consumption_plus_tax_policy = policy_c .+ consumption_tax_policy
    ss.consumption_tax_rate[i] = consumption_tax_policy ./ consumption_plus_tax_policy
    ss.consumption_tax_rate_vat[i] = consumption_tax_policy ./ policy_c
    
    # Gini coefficients
    ss.consumption_tax_gini[i] = compute_gini(consumption_tax_policy, stat_dist; plot_curve = false)
    ss.consumption_base_gini[i] = compute_gini(consumption_plus_tax_policy, stat_dist; plot_curve = false)
    ss.consumption_gini[i] = compute_gini(policy_c, stat_dist; plot_curve = false)

    # === Consumption distribution stats ===    
    ss.cons_dist_stats[i] = compute_income_distribution_stats(stat_dist, policy_c)

    avg_cons_stats = compute_average_income_stats(stat_dist, policy_c; cutoffs = [0.5, 0.9, -0.1])

    ss.t10tob50_cons_ratio[i] = avg_cons_stats[3][2] / avg_cons_stats[1][2]
    ss.t10tob90_cons_ratio[i] = avg_cons_stats[3][2] / avg_cons_stats[2][2]


    # Kakwani Index
    ss.kakwani_cons_tax[i] = ss.consumption_tax_gini[i] - ss.consumption_base_gini[i]

    # === Average effective rates and AETR ===
    ss.avg_rates_Ctax[i] = compute_average_rate_stats(stat_dist, consumption_tax_policy, policy_l_incpt = consumption_plus_tax_policy,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    ss.avg_rates_Ctax_vat[i] = compute_average_rate_stats(stat_dist, consumption_tax_policy, policy_l_incpt = policy_c,
                           cutoffs = [0.1, 0.5, 0.9, -0.1])
    ss.b10t10_rates_Ctax[i] = round.((ss.avg_rates_Ctax_vat[i][1][2], ss.avg_rates_Ctax_vat[i][4][2]), digits = 3)

    
end

# compute_average_rate_stats(stat_dist, labor_tax_policy, policy_l_incpt = policy_l_incpt,
#                            cutoffs = [0.1, 0.5, 0.9, -0.1])

# Get extrema for consumption tax rates
ss.cons_tax_rates_min = [round.(t; digits=3) for t in minimum.(ss.consumption_tax_rate_vat)]
ss.cons_tax_rates_max = [round.(t; digits=3) for t in maximum.(ss.consumption_tax_rate_vat)]
ss.cons_tax_rates_extrema = [round.(t; digits=3) for t in extrema.(ss.consumption_tax_rate_vat)]


# Compute % variation from the baseline for each column
ss_pct = DataFrame()
float_vars = ["lambda_y", "tau_y", "lambda_c", "tau_c", "tau_k", 
              "aggC", "aggG", "aggH", "aggK", "aggL", "aggT_c", 
              "aggT_k", "aggT_y", "aggY", "r", "w"]

for col in float_vars
    baseline_val = baseline[1, col]
    ss_pct[!, col] = 100 .* (ss[!, col] .- baseline_val) ./ baseline_val
end

# Add % Variation for progressivity parameters also to main dataframe
for col in ["tau_y", "tau_c"]
    baseline_val = baseline[1, col]
    new_col = "delta_" * col
    ss[!, new_col] = 100 .* (ss[!, col] .- baseline_val) ./ baseline_val
end 

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
                      low = Any[], 
                      mid = Any[], 
                      high = Any[])

#----# AGGREGATE TABLE SECTION 1 - TAXES #----#
sec1_table = deepcopy(agg_table)

# Add Government expenditure and tau_y, tau_c values in first and second, third rows
push!(sec1_table, extract_low_mid_high(ss, :aggG; 
                                      table_string = "Government Expenditure",
                                      normalise = "num"))
                                    
push!(sec1_table, extract_low_mid_high(ss, :tau_y; 
                                      table_string = "Labor Income Tax Progressivity"))
push!(sec1_table, extract_low_mid_high(ss, :tau_c; 
                                      table_string = "Consumption Tax Progressivity"))

# Add Taxes bottom and top 10% average effective rates
push!(sec1_table, extract_low_mid_high(ss, :b50t10aetr_Wtax; table_string = "AETR (%) - Labor Income - Bottom 50%, Top 10%", as_percentage = true))
push!(sec1_table, extract_low_mid_high(ss, :b10t10_rates_Ctax; table_string = "AETR (%) - Consumption — Bottom 10%, Top 10%", as_percentage = true))
push!(sec1_table, extract_low_mid_high(ss, :cons_tax_rates_extrema; table_string = "Consumption Tax Effective Rates (%) - Extrema", as_percentage = true))

# Add Revenue shares 
ss.TRevC = ss.aggT_c ./ ss.aggG
ss.TRevW = ss.aggT_y ./ ss.aggG
ss.TRevK = ss.aggT_k ./ ss.aggG

push!(sec1_table, extract_low_mid_high(ss, :TRevW; 
                                      table_string = "Labor Income Taxes - % of Revenue",
                                      as_percentage = true))
push!(sec1_table, extract_low_mid_high(ss, :TRevC; 
                                      table_string = "Consumption Taxes - % of Revenue",
                                      as_percentage = true))
push!(sec1_table, extract_low_mid_high(ss, :TRevK; 
                                      table_string = "Capital Return Taxes - % of Revenue",
                                      as_percentage = true))


# Plot and add main aggregates to DataFrame for table
#----# AGGREGATE TABLE SECTION 2 - AGGREGATES #----#
sec2_table = deepcopy(agg_table)

# Output
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggY;
                     ylabel = "Δ Aggregate Output (%)",
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/aggY_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :aggY; table_string = "Agg. Output"))

# Consumption                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggC;
                     ylabel = "Δ Aggregate Consumption (%)", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/aggC_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :aggC; table_string = "Agg. Consumption"))

# Capital                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggK;
                     ylabel = "Δ Aggregate Capital (%)", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/aggK_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :aggK; table_string = "Agg. Capital"))

# Hours worked                                        
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.aggH;
                     ylabel = "Δ Aggregate Hours Worked (%)", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/aggH_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :aggH; table_string = "Agg. Hours Worked"))

# Prices
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.r;
                     ylabel = "Δ Interest Rate (%)", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/r_sims.png")

plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.r .* 100;
                     ylabel = "Interest Rate (%)",  
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/r_abs_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :r; table_string = "Interest Rate"))
            
plot_colored_scatter(ss_pct.tau_y, ss_pct.tau_c, ss_pct.w;
                     ylabel = "Δ Equilibrium Wage (%)", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/w_sims.png")

push!(sec2_table, extract_low_mid_high(ss, :w; table_string = "Wage"))

#-----# Inequalities #-----#
#----# AGGREGATE TABLE SECTION 2 - DISTRIBUTIONS #----#
sec3_table = deepcopy(agg_table)

# Income
# Gini for pre-tax income
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.gini_income;
                     ylabel = "Pre-Tax Income - Gini Coefficient", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/gini_ptinc_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :gini_income; table_string = "Gini Coefficient for Pre-Tax Income"))

plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.gini_disp_income;
                     ylabel = "After-Tax Income - Gini Coefficient", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/gini_atinc_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :gini_disp_income; table_string = "Gini Coefficient for After-Tax Income"))

# Consumption
# Gini for pre-tax income
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.consumption_gini;
                     ylabel = "Consumption - Gini Coefficient", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/gini_cons_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :consumption_gini; table_string = "Gini Coefficient for Consumption"))


# T10/B50 Pre-Tax Average Income Ratio
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.t10tob50_ptinc_ratio;
                     ylabel = "T10/B50 Average Pre-Tax Income Ratio", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/t10b50_inc_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :t10tob50_ptinc_ratio; table_string = "Pre-Tax Income - T10/B50 Average Ratio"))

# T10/B50 After-Tax Average Income Ratio
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.t10tob50_atinc_ratio;
                     ylabel = "T10/B50 Pre-Tax Income Share Ratio", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/t10b50_atinc_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :t10tob50_atinc_ratio; table_string = "After-Tax Income - T10/B50 Average Ratio"))

# T10/B50 Average Consumption Ratio 
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.t10tob50_cons_ratio;
                     ylabel = "T10/B50 Average Consumption Ratio", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/t10b50_cons_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :t10tob50_cons_ratio; table_string = "Consumption - T10/B50 Average Ratio"))


#-----# Taxes #-----#
# Consumption
# Kakwani Consumption Tax
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.kakwani_cons_tax;
                     ylabel = "Kakwani Index for Consumption Tax", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/Ctax_Kakwani_sims.png")

# push!(sec1_table, extract_low_mid_high(ss, :kakwani_cons_tax; table_string = "Kakwani Index for Consumption Tax"))

# Min and max Consumption Tax Rates
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.cons_tax_rates_max .- ss.cons_tax_rates_min;
                     ylabel = "Consumption Tax Rate Range Width", 
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/Ctax_range_width_sims.png")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------------# 4. ANALYSING DISTRIBUTIONS #------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Analyse distributional effects - who gained, who lost?
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------------------# Consumption #-------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Plot
densC = build_density_dict(ss, :distC, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densC, [:low, :mid, :high];
                        ylabel = "Consumption (c)", title = "Consumption Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Compare distributions across tax regimes
decC = Dict(:low => ss.cons_deciles[idx_low_tau_y], 
            :mid => ss.cons_deciles[idx_middle], 
            :high => ss.cons_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decC,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Consumption Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:red, :gray75, :blue],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/cons_dist_by_decile.png", p)

# Compare differences 
decCdiff = Dict(:lowdiff => ss.cons_deciles[idx_low_tau_y] - ss.cons_deciles[idx_middle],
                :highdiff => ss.cons_deciles[idx_high_tau_y] - ss.cons_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decCdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lb,
    title = "Consumption Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:red, :blue],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/cons_diff_dist_by_decile.png", p)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# Capital #---------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

densK = build_density_dict(ss, :distK, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densK, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Wealth Held (density)", title = "Asset distribution",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Compare distributions across tax regimes
ss.asset_deciles = compute_decile_distribution.(ss.stat_dist, ss.policy_a)

decK = Dict(:low => ss.asset_deciles[idx_low_tau_y], 
            :mid => ss.asset_deciles[idx_middle], 
            :high => ss.asset_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decK,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Net Wealth Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:yellow2, :gray75, :mediumpurple2],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/asset_dist_by_decile.png", p)

# Compare differences 
decKdiff = Dict(:lowdiff => ss.asset_deciles[idx_low_tau_y] - ss.asset_deciles[idx_middle],
                :highdiff => ss.asset_deciles[idx_high_tau_y] - ss.asset_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decKdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lt,
    title = "Net Wealth Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:yellow2, :mediumpurple2],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/asset_diff_dist_by_decile.png", p)

# Compute Gini for wealth distribution 
ss.wealth_gini = similar(ss.lambda_c)
for i in 1:nrow(ss)
    ss.wealth_gini[i] = compute_gini(ss.policy_a[i], ss.stat_dist[i])
end

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-------------------------------# Hours worked #------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
 
densH = build_density_dict(ss, :distH, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densH, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Wealth Held (density)", title = "Asset distribution",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (40, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Compare distributions across tax regimes
ss.hours_deciles = compute_decile_distribution.(ss.stat_dist, ss.policy_l)

decH = Dict(:low => ss.hours_deciles[idx_low_tau_y], 
            :mid => ss.hours_deciles[idx_middle], 
            :high => ss.hours_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decH,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Labor (Hours) Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:darkorange1, :gray75, :limegreen],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/hours_dist_by_decile.png", p)

# Compare differences 
decHdiff = Dict(:lowdiff => ss.hours_deciles[idx_low_tau_y] - ss.hours_deciles[idx_middle],
                :highdiff => ss.hours_deciles[idx_high_tau_y] - ss.hours_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decHdiff,
    [:lowdiff, :highdiff];
    legend_pos = :rt,
    title = "Labor (Hours) Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:darkorange1, :limegreen],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/hours_diff_dist_by_decile.png", p)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------------------------------# Taxes #----------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Plot average and marginal rates for labor income taxes
# Plot AER 
aer_b(y) = (1 - ss.lambda_y[idx_middle] * y ^ (-ss.tau_y[idx_middle])) *100
aer_acp(y) = (1 - ss.lambda_y[idx_low_tau_y] * y ^ (-ss.tau_y[idx_low_tau_y])) *100
aer_alp(y) = (1 - ss.lambda_y[idx_high_tau_y] * y ^ (-ss.tau_y[idx_high_tau_y])) *100

p = plot_function_family([aer_b, aer_acp, aer_alp], 
    ["Baseline", "ACP", "ALP"], 
    minimum(ss.policy_l_incpt[idx_low_tau_y]), maximum(ss.policy_l_incpt[idx_low_tau_y]);
    laby = "AER (%)",
    labx = "Pre-Tax Income",
    cmap = :Set1_8,
    # y_low = aer_Crev_eq(minimum(policy_c_b)), 
    # y_low = -56.0,
    y_up = 45.0,
    size = (500, 500),
    y_ticks = [0.0, 10.0, 20.0, 30.0, 40.0]
)

CairoMakie.save("output/figures/equivalent_regimes/aers_acp_alp.png", p)


# Plot AMR across consumption taxes 
amr_b(y) = (1 - ss.lambda_y[idx_middle] * (1-ss.tau_y[idx_middle]) * y ^ (-ss.tau_y[idx_middle])) *100
amr_acp(y) = (1 - ss.lambda_y[idx_low_tau_y] * (1-ss.tau_y[idx_low_tau_y]) * y ^ (-ss.tau_y[idx_low_tau_y])) *100
amr_alp(y) = (1 - ss.lambda_y[idx_high_tau_y] * (1-ss.tau_y[idx_high_tau_y]) * y ^ (-ss.tau_y[idx_high_tau_y])) *100

p = plot_function_family([amr_b, amr_acp, amr_alp], 
    ["Baseline", "ACP", "ALP"], 
    minimum(ss.policy_l_incpt[idx_low_tau_y]), maximum(ss.policy_l_incpt[idx_low_tau_y]);
    laby = "MR (%)",
    labx = "Pre-Tax Income",
    cmap = :Set1_8,
    size = (500, 500),
    y_ticks = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    leg_pos = :rb
)

CairoMakie.save("output/figures/equivalent_regimes/mrs_acp_alp.png", p)



#----# Labor Taxes #----#
densWtax = build_density_dict(ss, :distWtax, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densWtax, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Labor Income Taxes", title = "Labor Income Tax Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (1, 200),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

# Compare distributions across tax regimes
ss.Wtax_deciles = compute_decile_distribution.(ss.stat_dist, ss.labor_tax_policy)

decWtax = Dict(:low => ss.Wtax_deciles[idx_low_tau_y], 
            :mid => ss.Wtax_deciles[idx_middle], 
            :high => ss.Wtax_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decWtax,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Labor Income Tax Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:red, :gray75, :blue],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxW_dist_by_decile.png", p)

# Compare differences 
decWtaxdiff = Dict(:lowdiff => ss.Wtax_deciles[idx_low_tau_y] - ss.Wtax_deciles[idx_middle],
                :highdiff => ss.Wtax_deciles[idx_high_tau_y] - ss.Wtax_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decWtaxdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lt,
    title = "Labor Income Tax Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:red, :blue],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxW_diff_dist_by_decile.png", p)

#----# Consumption taxes #----#
densCtax = build_density_dict(ss, :distCtax, idx_low_tau_y, idx_middle, idx_high_tau_y)
plot_densities_by_group(densCtax, [:low, :mid, :high];
                        xlabel = "Wealth level", ylabel = "Consumption Taxes", title = "Consumption Tax Distribution by Wealth Level",
                        x_vec = a_grid, cmap = :avocado,
                        index_range = (1, 240),
                        leg_labels = ["Low Labor Income Tax Progressivity", "Baseline", 
                                        "High Labor Income Tax Progressivity"]
                        )

                        
# Compare distributions across tax regimes
ss.Ctax_deciles = compute_decile_distribution.(ss.stat_dist, ss.consumption_tax_policy)

decCtax = Dict(:low => ss.Ctax_deciles[idx_low_tau_y], 
            :mid => ss.Ctax_deciles[idx_middle], 
            :high => ss.Ctax_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decCtax,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Consumption Tax Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:brown3, :gray75, :forestgreen],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxC_dist_by_decile.png", p)

# Compare differences 
decCtaxdiff = Dict(:lowdiff => ss.Ctax_deciles[idx_low_tau_y] - ss.Ctax_deciles[idx_middle],
                :highdiff => ss.Ctax_deciles[idx_high_tau_y] - ss.Ctax_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decCtaxdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lt,
    title = "Consumption Tax Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:brown3, :forestgreen],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxC_diff_dist_by_decile.png", p)

#----# Capital Taxes #----#
ss.capital_tax_policy = similar(ss.labor_tax_policy)
for i in 1:nrow(ss)
    # Extract capital_return policy
    ss.capital_tax_policy[i] = ss.policy_a[i] .* ss.r[i] .* ss.tau_k[i]
end

# Compare distributions across tax regimes
ss.Ktax_deciles = compute_decile_distribution.(ss.stat_dist, ss.capital_tax_policy)

decKtax = Dict(:low => ss.Ktax_deciles[idx_low_tau_y], 
            :mid => ss.Ktax_deciles[idx_middle], 
            :high => ss.Ktax_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decKtax,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Capital Tax Distribution by Decile",
    ylabel = "Share (%)",
    bar_palette = [:yellow2, :gray75, :mediumpurple2],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxK_dist_by_decile.png", p)

# Compare differences 
decKtaxdiff = Dict(:lowdiff => ss.Ktax_deciles[idx_low_tau_y] - ss.Ktax_deciles[idx_middle],
                :highdiff => ss.Ktax_deciles[idx_high_tau_y] - ss.Ktax_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decKtaxdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lt,
    title = "Capital Tax Distribution by Decile - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:yellow2, :mediumpurple2],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxK_diff_dist_by_decile.png", p)


#----# All Taxes - Who pays more, who pays less? #----#

ss.tax_policy = ss.capital_tax_policy .+ ss.labor_tax_policy .+ ss.consumption_tax_policy

# Compare distributions across tax regimes
ss.Tax_deciles = compute_decile_distribution.(ss.stat_dist, ss.tax_policy)

decTax = Dict(:low => ss.Tax_deciles[idx_low_tau_y], 
            :mid => ss.Tax_deciles[idx_middle], 
            :high => ss.Tax_deciles[idx_high_tau_y])

p = plot_decile_distributions_by_group(decTax,
    [:low, :mid, :high];
    legend_pos = :lt,
    title = "Tax Distribution by Decile - All Taxes",
    ylabel = "Share (%)",
    bar_palette = [:orange2, :gray75, :limegreen],
    leg_labels = ["Aug. Consumption Progressivity", "Baseline", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxesAll_dist_by_decile.png", p)

# Compare differences 
decTaxdiff = Dict(:lowdiff => ss.Tax_deciles[idx_low_tau_y] - ss.Tax_deciles[idx_middle],
                :highdiff => ss.Tax_deciles[idx_high_tau_y] - ss.Tax_deciles[idx_middle])
            
p = plot_decile_distributions_by_group(decTaxdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lt,
    title = "Tax Distribution by Decile - All Taxes - % Difference with Baseline",
    ylabel = "Share (%)",
    bar_palette = [:orange2, :limegreen],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"]
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/TaxesAll_diff_dist_by_decile.png", p)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------------# Welfare #---------------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# # Recompute value functions for all steady states
# ss.ep_valuef = similar(ss.stat_dist)

# for i in 1:nrow(ss)
#     @views ss.ep_valuef[i] = recompute_value(ss.policy_c[i], ss.policy_l[i], ss.policy_a[i], pi_rho;
#                                            a_grid, gpar, hhpar, tol=1e-10, max_iter=10_000)
# end

# baseline.ep_valuef = similar(baseline.stat_dist)
# baseline.ep_valuef[1] = recompute_value(baseline.policy_c[1], baseline.policy_l[1], 
#                              baseline.policy_a[1], pi_rho;
#                              a_grid, gpar, hhpar, tol=1e-10, max_iter=10_000)

# plot_value_function(baseline.ep_valuef[1], a_grid, rho_grid; taxes = baseline.taxes[1])

# Compute cev and aggCEV for all steady states
ss.cev = similar(ss.stat_dist)
ss.aggCEV = similar(ss.aggG)
ss.cev_deciles = similar(ss.cons_deciles)
ss.cev_by_inc_decile = similar(ss.cons_deciles)

for i in 1:nrow(ss) 
    ss.cev[i], ss.aggCEV[i] = compute_cev(ss.valuef[i], baseline.valuef[1], baseline.policy_c[1], 
            baseline.policy_a[1], baseline.stat_dist[1], pi_rho)
    
    ss.cev_deciles[i] = compute_decile_distribution(ss.stat_dist[i], ss.cev[i])

    ss.cev_by_inc_decile[i] = compute_avg_cevs_per_income_decile(ss.cev[i], ss.policy_l_incpt[idx_middle], ss.stat_dist[i]) ./ 10

end


# Plot aggregate CEV
plot_colored_scatter(ss.delta_tau_y, ss.delta_tau_c, ss.aggCEV;
                     ylabel = "Aggregate CEV (%)",
                     cmap = :avocado,
                     save_plot = true, 
                     save_path = "output/figures/equivalent_regimes/aggregates/aggCEV_sims.png")

push!(sec3_table, extract_low_mid_high(ss, :aggCEV; table_string = "CEV (%)"))


# Compare CEV distributions across tax regimes - Who gained, who lost?
# Compare differences 
decCEVdiff = Dict(:lowdiff => ss.cev_by_inc_decile[idx_low_tau_y],
                :highdiff => ss.cev_by_inc_decile[idx_high_tau_y])
            
p = plot_decile_distributions_by_group(decCEVdiff,
    [:lowdiff, :highdiff];
    legend_pos = :lb,
    title = "CEV by Income Decile - % Difference with Baseline",
    ylabel = "Δ (%)",
    bar_palette = [:red, :blue],
    leg_labels = ["Aug. Consumption Progressivity", "Aug. Labor Progressivity"],
    as_percentage = false
    )

CairoMakie.save("output/figures/equivalent_regimes/distributions/cev_dist_by_decile.png", p)

# Add CEV to table 
push!(sec2_table, extract_low_mid_high(ss, :aggCEV; table_string = "CEV (%)"))


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------# 4. FINALISE OUTPUT #----------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

agg_table = vcat(sec1_table, sec2_table, sec3_table)

# Output in LaTeX
at_tek = pretty_table(agg_table, backend = Val(:latex))

open("output/tables/equivalent_regimes/tax_summary.tex", "w") do io
    pretty_table(io, sec1_table, backend = Val(:latex))
end

# Round before outputting
push!(sec3_table, extract_low_mid_high(ss, :wealth_gini; 
                                       table_string = "Gini Coefficient for Wealth"))
sec3_table.low .= round.(sec3_table.low, digits=3)
sec3_table.mid .= round.(sec3_table.mid, digits=3)
sec3_table.high .= round.(sec3_table.high, digits=3)
open("output/tables/equivalent_regimes/inequality_welfare.tex", "w") do io
    pretty_table(io, sec3_table, backend = Val(:latex))
end

open("output/tables/equivalent_regimes/agg_table.tex", "w") do io
    pretty_table(io, agg_table, backend = Val(:latex))
end

