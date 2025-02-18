###############################################################################
############################### PARAMETERS.JL #################################

############## This script defines and useful functions to solve ##############
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

# # Functions to save/read matrix in a txt file 

function SaveMatrix(matrix, filepath::String)
    open(filepath, "w") do file
        # Write the dimensions of the matrix as the first line
        dims_str = join(size(matrix), ",")
        write(file, dims_str * "\n")
        # Write a newline to separate dimensions from data
        # write(file, "\n")
        # Write the matrix data
        writedlm(file, matrix, ',')
    end
    return print("Matrix saved in $filepath")
end

function ReadMatrix(filepath::String)
    local matrix
    open(filepath, "r") do file
        # Read the dimensions from the first line
        dims = parse.(Int, split(readline(file), ','))
        println("Dimensions read from file: ", dims)

        # Read the matrix data
        matrix = readdlm(file, ',', Float64)

        # Reshape the data to the original dimensions
        matrix = reshape(matrix, Tuple(dims))
    end
    return matrix
end
