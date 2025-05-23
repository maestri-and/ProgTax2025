###############################################################################
########################### AUXILIARYFUNCTIONS.JL #############################

################ This script defines useful functions to solve ################
###################### the benchmark ProgTax(2025) model ######################

###############################################################################


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------# 0. IMPORTING LIBRARIES AND DEFINING EXPORTS #---------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

using LinearAlgebra
using DelimitedFiles
using DataFrames


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#---------------------------# 1. GENERATING GRIDS #---------------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# Function to generate grid with different spacing 
# Currently supported: uniform (equally spaced), log-spaced, polynomial, chebyshev (cosine)
function makeGrid(xmin, xmax, n_values; 
                  grid_type = "uniform", cbsv_alpha = 1.0, pol_power = 4,
                  lab_double_bounds = (0.0, 0.6))

    if grid_type == "uniform"
        return collect(range(xmin, length=n_values, stop=xmax))

    elseif grid_type == "log"
        # Construct log-spaced grid
        # Correct for negative boundary if necessary 
        shift = xmin > 0 ? 0 : abs(xmin) + 0.01
        # Computed corrected logs 
        l_xmin = log(xmin + shift)
        l_xmax = log(xmax + shift)
        # Take uniformly spaced grid over logs
        log_grid = collect(range(l_xmin, length=n_values, stop=l_xmax))
        # Apply inverse transformation to obtain log-spaced grid, re-correcting if needed 
        return ℯ.^(log_grid) .- shift

    elseif grid_type == "polynomial"
        # Transformation: x(g) = a + (b - a) * g^γ, with g∈[0,1]
        # Consider equally spaced grid on [0,1] of length n_values
        unif_grid = range(0, 1, n_values)
        # Compute grid for given gamma 
        return [xmin + (xmax - xmin) * g^pol_power for g in unif_grid]

    elseif grid_type == "chebyshev"
        return [(xmin + xmax)/2 + (xmax - xmin)/2 * sign(cos(π * (1 - i/(n_values - 1)))) * abs(cos(π * (1 - i/(n_values - 1))))^cbsv_alpha for i in 0:(n_values-1)]    

    elseif grid_type == "labor-double"
        # For labor grid, double density within range lab_double_bounds
        lb_lo, lb_hi = lab_double_bounds
        @assert xmin <= lb_lo <= lb_hi <= xmax "lab_double_bounds must lie within [xmin, xmax]"

        # Lengths of regions
        L_out = (lb_lo - xmin) + (xmax - lb_hi)
        L_in  = lb_hi - lb_lo

        # Total "effective" length in d_in units:
        # Outer counts as L_out / (2*d_in), inner as L_in / d_in
        # So total points ≈ L_out / (2*d) + L_in / d + 1 (include endpoint)
        # Solve for d
        d = (2 * L_in + L_out) / (2 * (n_values - 1))

        # Number of points in each region
        n_left  = round(Int, (lb_lo - xmin) / (2d))
        n_inner = round(Int, (lb_hi - lb_lo) / d)
        n_right = n_values - 1 - n_left - n_inner  # -1 for endpoint adjustment

        g_left  = range(xmin, lb_lo, length=n_left + 1)[1:end-1]
        g_inner = range(lb_lo, lb_hi, length=n_inner + 1)[1:end-1]
        g_right = range(lb_hi, xmax, length=n_right + 1)

        return collect(vcat(g_left, g_inner, g_right))
    else
        @error("Grid type not supported!")
        error("Grid type not supported!")
    end
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------# 2. EXPORTING AND IMPORTING MODEL RESULTS #----------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Exporting results session detail 
function print_simulation_details(filepath::String; 
                                  session_time = session_time,
                                  grid_parameters = gpar,
                                  asset_grid_type = a_gtype,
                                  hhpar = hhpar,
                                  fpar = fpar,
                                  comp_params =comp_params
                                  )
    # Write file
    open(filepath, "w") do file
        
        # Headline
        write(file, "BASELINE MODEL SIMULATION - DETAILS" * "\n")
        write(file, "\n")

        # Write grid details  
        write(file, "Grids - Details" * "\n")
        write(file, "Asset grid type: $(a_gtype)" * "\n") 
        write(file, "Grid parameters: $(gpar)" * "\n")
        write(file, "\n")

        # Write parameters details
        write(file, "Structural Parameters - Details" * "\n")
        write(file, "Taxes parameters: $(taxes)" * "\n")
        write(file, "Household parameters: $(hhpar)" * "\n")
        write(file, "Firm parameters: $(fpar)" * "\n")
        write(file, "Computational parameters: $(comp_params)" * "\n")
        write(file, "\n")
        
        # Write time details
        write(file, "Time spent: $(session_time)")
    end
    return nothing
end



# Exporting model results: steady states by tax struct
function SaveMatrix(matrix, filepath::String; overwrite=false, write_parameters = true, taxes=taxes)
    # Append or overwrite if specified
    mode = overwrite ? "w" : "a"
    open(filepath, mode) do file
        # Add an empty line to divide different outputs
        if !overwrite && isfile(filepath) && filesize(filepath) > 0
            write(file, "\n")  
        end
        
        # Write taxation parameters 
        if write_parameters
            write(file, "$(taxes)" * "\n")
        end

        # Add size for matrices
        if matrix isa AbstractArray
            dims_str = join(size(matrix), ",")
            write(file, dims_str * "\n")
        elseif matrix isa AbstractFloat
            write(file, "1" * "\n")
        elseif isstructtype(typeof(matrix))
            write(file, "-99" * "\n")
            # Extract tuple
            matrix = [getfield.(Ref(matrix), fieldnames(typeof(matrix)))]
        else
            @error("Wrong argument type! Ensure your input is a matrix or a float!")
            error("Wrong argument type! Ensure your input is a matrix or a float!")
        end

        # Write element
        writedlm(file, matrix, ',')
    end
    @info("Matrix saved in $filepath")
    return nothing
end

# Export Calibration Table to Latex
function output_parameters_latex(calibration_pars::Vector{String}, 
                                 values::Vector{Float64}, 
                                 save_path::String)

    # Build dataframe
    df = DataFrame(param = calibration_pars, value = round.(values, digits=4))

    # Updated label map
    latex_names = Dict(
        "hhpar.beta"        => raw"$\beta$ - Discount factor",
        "hhpar.rra"         => raw"$\sigma$ - Relative risk aversion",
        "hhpar.dis_labor"   => raw"$B$ - Disutility of labor",
        "hhpar.inv_frisch"  => raw"$\psi$ - Inverse of Frisch elasticity",
        "rhopar.rho_prod_ar1"   => raw"$\rho_{AR(1)}$ - Prod. AR(1) Process persistency",
        "rhopar.sigma_prod_ar1" => raw"$\sigma_{AR(1)}$ - Prod. AR(1) Process volatility",
        "rhopar.n_prod_ar1"     => raw"N. States - Rouwenhorst discretisation",
        "fpar.alpha"        => raw"$\alpha$ - Capital share",
        "fpar.delta"        => raw"$\delta$ - Depreciation rate",
        "fpar.tfp"          => raw"$A$ - Total Factor Productivity",
        "taxes.lambda_y"    => raw"$\lambda_y$ - Labor tax scale",
        "taxes.tau_y"       => raw"$\tau_y$ - Labor tax progressivity",
        "taxes.lambda_c"    => raw"$\lambda_c$ - Consumption tax scale",
        "taxes.tau_c"       => raw"$\tau_c$ - Consumption tax progressivity",
        "taxes.tau_k"       => raw"$\tau_k$ - Capital return tax rate",
        "gpar.a_min"        => raw"$\underline{a}$ - Borrowing Limit"
    )

    # Filter
    df = filter(row -> row.param in keys(latex_names), df)

    # Relabel
    df.param = get.(Ref(latex_names), df.param, df.param)

    # Export LaTeX
    open(save_path, "w") do io
        pretty_table(io, df, header = ["Parameter", "Value"], backend = Val(:latex))
    end

    # Clean LaTeX ex-post
    lines = readlines(save_path)

    # cleaned = replace.(lines, "\\\\" => "\\")                # fix over-escaping
    cleaned = replace.(lines, raw"\$" => raw"$")              # fix \$ → $
    cleaned = replace.(cleaned, raw"\{" => raw"{")              # fix \$ → $
    cleaned = replace.(cleaned, raw"\}" => raw"}")
    cleaned = replace.(cleaned, raw"\textbackslash{}" => "\\")# remove \textbackslash{}
    cleaned = replace.(cleaned, raw"\_" => "_")             # fix escaped underscores

    open(save_path, "w") do io
        for line in cleaned
            println(io, line)
        end
    end

end


# Importing matrix printed to .txt - DEPRECATED 
function ReadMatrix(filepath::String)
    local matrix
    open(filepath, "r") do file
        # Read the dimensions from the first line
        dims = parse.(Int, split(readline(file), ','))
        @info("Dimensions read from file: ", dims)

        # Read the matrix data
        matrix = readdlm(file, ',', Float64)

        # Reshape the data to the original dimensions
        matrix = reshape(matrix, Tuple(dims))
    end
    return matrix
end

# Importing model results into dataframe
function read_results_from_txt(filepath::String)
    # Extract file name without extension for the data column
    colname = split(basename(filepath), ".")[1]

    # Initialise arrays
    # Recognise dimensions
    dims = Vector{Int}()
    open(filepath, "r") do io
        # Read first dimension line
        skipline = readline(io)             # Taxes line, ignore
        dims_loc = parse.(Int, split(readline(io), ','))
        for i in dims_loc
            push!(dims, i)
        end
    end
    
    # Initialise vector of values based on dims and taxes vectors
    values = Any[]
    lambda_y_vec = Float64[]
    tau_y_vec    = Float64[]
    lambda_c_vec = Float64[]
    tau_c_vec    = Float64[]
    tau_k_vec    = Float64[]
    tparams = zeros(5)

    # Retrieve data
    open(filepath, "r") do io
        while !eof(io)
            # -------- Line 1: TAX STRUCT --------
            # Read first line
            tax_line = readline(io)  # Example: "Taxes(0.7, 0.0, 0.7, 0.05, 0.2)"

            # Clean the line and extract the numbers using regex:
            tparams .= parse.(Float64, split(strip(tax_line[7:end-1]), ","))

            # Store value into respective columns
            push!(lambda_y_vec, tparams[1])
            push!(tau_y_vec,    tparams[2])
            push!(lambda_c_vec, tparams[3])
            push!(tau_c_vec,    tparams[4])
            push!(tau_k_vec,    tparams[5])


            # -------- Line 2: DIMENSIONS --------
            # Skip as we already read dims at the beginning
            skipline = readline(io)  # Example: "1" or "7,100"

            # -------- Line 3: VALUE --------
            # Read third line and store according to type
            if dims == [1]
                # It's a scalar float
                push!(values, parse(Float64, readline(io)))
            elseif dims == [-99]
                # It's a struct - read tuple
                line = readline(io)
                vals = parse.(Float64, split(line, ','))
                push!(values, Tuple(vals))
            else
                # It's a matrix stored as a comma-separated line
                rows = Int(dims[1])
                row = [parse.(Float64, split(readline(io), ',')) for _ in 1:rows]
                push!(values, hcat(row...)')
            end

            # -------- Line 4: EMPTY LINE --------
            # Skip the blank separator between entries
            if !eof(io)
                readline(io)
            end
        end
    end

    # Construct DataFrame
    df = DataFrame(lambda_y = lambda_y_vec,    
                   tau_y = tau_y_vec,
                   lambda_c = lambda_c_vec,
                   tau_c = tau_c_vec,
                   tau_k = tau_k_vec)

    # Add values column
    df[!, Symbol(colname)] = values
    return df
end

# Build a dataframe with all model results by tax struct
function get_model_results(folderpath::String; ignorefiles = ["placeholder.txt", "search_results.txt"], 
                           merging_keys = [:lambda_y, :tau_y, :lambda_c, :tau_c, :tau_k])
    # This function constructs a full DataFrame from .txt files stored
    # in the folder containing model results 

    # Get files to be read - ignorefiles is a vector containing
    # files not to be read in 
    datafiles = setdiff(readdir(folderpath), ignorefiles)

    # Construct first dataframe
    df = read_results_from_txt(joinpath(folderpath,datafiles[1]))
    # Remove duplicates
    df = unique(df)

    # Iterate over files and merge
    for file in datafiles[2:end]
        try
            # Read
            temp = read_results_from_txt(joinpath(folderpath, file))

            # Remove duplicates
            temp = unique(temp)

            # Merge 
            df = outerjoin(df, temp, on = merging_keys)
        catch e
            @error("Error in file: $file")
            throw(e)
        end
    end

    return df
end 

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------# 3. EXPORTING AND IMPORTING TAX REGIME SEARCH RESULTS #----------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# Exporting search results: equivalent regime, equilibrium rate, aggregate G
function WriteTaxSearchResults(eq_taxes, eq_r, eq_exp, filepath::String)
    line = string(eq_taxes, ";", eq_r, ";", eq_exp)
    write(filepath, line)
    @info("[Thread $(Threads.threadid())] Tax Search Results saved in $filepath")
    return nothing
end

function print_eq_regime_search_session_details(regimes, eq_rates, taxes, filepath::String; 
    session_time = session_time,
    grid_parameters = gpar,
    asset_grid_type = a_gtype,
    hhpar = hhpar,
    fpar = fpar,
    comp_params =comp_params,
    regime_search_only = false)
    # Write file
    open(filepath, "w") do file

        # Write time details
        if regime_search_only
            write(file, "TAX REGIME SEARCH SESSION - DETAILS" * "\n")
        else
            write(file, "EQUIVALENT TAX REGIME SIMULATIONS - DETAILS" * "\n")
        end
        write(file, "Time spent: $(session_time)" * "\n")
        write(file, "\n")

        # Write parameters details
        write(file, "Target regime: $(taxes)" * "\n")
        l = length(regimes)
        write(file, "Found $l equivalent regimes:" * "\n")
        for i in 1:l
            write(file, "$(regimes[i]);$(eq_rates[i])\n")
        end
        write(file, "\n")
        write(file, "Baseline taxes parameters: $(taxes)" * "\n")
        write(file, "Household parameters: $(hhpar)" * "\n")
        write(file, "Firm parameters: $(fpar)" * "\n")
        write(file, "Computational parameters: $(comp_params)" * "\n")

        # Write grid details  
        write(file, "Asset grid type: $(a_gtype)" * "\n") 
        write(file, "Grid parameters: $(gpar)" * "\n")
    end
    return nothing
end

# Re-importing search results
function ReadTaxSearchResults(filepath::String)
    line = readline(filepath)
    parts = split(line, ';')
    return reshape(parts, 1, 3)  # 1×3 Array{String,2}
end

function ImportEquivalentTaxRegimes(root::String)
    # Returns a DataFrame with columns: tax_regime, r, aggG, filepath
    results = String[]

    for (dir, _, files) in walkdir(root)
        for f in files
            if f == "search_results.txt"
                push!(results, joinpath(dir, f))
            end
        end
    end

    df = DataFrame(tax_regime = Taxes[], r = Float64[], aggG = Float64[], filepath = String[])

    for rpath in results
        vals = split(readline(rpath), ';')
        # Evaluate Taxes struct
        tax = eval(Meta.parse(vals[1]))
        push!(df, (tax, parse(Float64, vals[2]), parse(Float64, vals[3]), rpath))
    end

    return df
end