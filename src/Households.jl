
# Create a struct for households
struct hh
    wealth::Float64 #Assets held
    prod::Float64   #Labor productivity 
    # age - TBM
end

# Define household utility 
function get_utility_hh(consumption, labor, rra, phi, frisch; normalise = false)
    # Compute households' utility - with normalisation if necessary 
    if normalise == true
        return (consumption ^ (1 - rra) - 1)/(1 - rra) - phi * (labor ^ (1 + frisch) - 1)/(1 + frisch)
    else
        return (consumption ^ (1 - rra))/(1 - rra) - phi * (labor ^ (1 + frisch))/(1 + frisch)
    end
end


# Define household taxes 
function tax_hh(z, lambda_z, tau_z)
    # Temporary: taxes only on wage
    return tax = z - lambda_z * z ^ (1 - tau_z)
end

function total_tax_hh(consumption_tax_hh, wage_tax_hh)
    # Temporary: taxes only on wage
    return total_tax = consumption_tax_hh + wage_tax_hh
end




