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



# Function to generate grid - TBM (can be adjusted as in Ragot et al. 2024)
function makeGrid(xmin, xmax, n_values)
    return collect(range(xmin, length=n_values, stop=xmax))
end

function SaveMatrix(matrix, filepath::String; overwrite=false, write_parameters = true, taxes=taxes)
    # Append or overwrite if specified
    mode = overwrite ? "w" : "a"
    open(filepath, mode) do file
        # Add an empty line to divide different outputs
        if !overwrite && isfile(filepath)
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
