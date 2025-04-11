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
function makeGrid(xmin, xmax, n_values; grid_type = "uniform", cbsv_alpha = 1.0, pol_power = 2)
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
    else
        @error("Grid type not supported!")
    end
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-----------------# 1. EXPORTING AND IMPORTING MODEL RESULTS #----------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

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
        else
            @error("Wrong argument type! Ensure your input is a matrix or a float!")
        end

        # Write element
        writedlm(file, matrix, ',')
    end
    @info("Matrix saved in $filepath")
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
    values = dims == [1] ? Float64[] : Array{Float64}[]
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
function get_model_results(folderpath::String; ignorefiles = ["placeholder.txt"], 
                           merging_keys = [:lambda_y, :tau_y, :lambda_c, :tau_c, :tau_k])
    # This function constructs a full DataFrame from .txt files stored
    # in the folder containing model results 

    # Get files to be read - ignorefiles is a vector containing
    # files not to be read in 
    datafiles = setdiff(readdir(folderpath), ignorefiles)

    # Construct first dataframe
    df = read_results_from_txt(joinpath(folderpath,datafiles[1]))

    # Iterate over files and merge
    for file in datafiles[2:end]
        try
            # Read
            temp = read_results_from_txt(joinpath(folderpath, file))

            # Merge 
            df = outerjoin(df, temp, on = merging_keys)
        catch e
            @error("Error in file: $file")
            throw(e)
        end
    end

    return df
end 