
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

# Compute consumption expenditure - RHS of budget constraint 
function get_Cexp(rho, w, r, l, a, a_prime, taxes)
    return taxes.lambda_y * (rho * w * l)^(1 - taxes.tau_y) + (1 + (1 - taxes.tau_k) * r) * a - a_prime
end

# Compute optimal labor - from analytical solution 
function get_opt_labor_from_FOC(c, rho, w, taxes, hh_parameters)
    l_star = ((taxes.lambda_y * (1 - taxes.tau_y) * (rho * w)^(1 - taxes.tau_y)) / #Numerator
             (2 - taxes.lambda_c * (1 - taxes.tau_c) * c ^ (-taxes.tau_c) * hh_parameters.phi * c^hh_parameters.rra) #Denominator
             ) ^ (1 / (hh_parameters.frisch + taxes.tau_y)) # Exponent

    return l_star
end


