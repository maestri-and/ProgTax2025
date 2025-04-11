
# Create a struct for households
struct hh
    wealth::Float64 #Assets held
    prod::Float64   #Labor productivity 
    # age - TBM
end

# Define household utility 
function get_utility_hh(consumption, labor, hhpar; normalise = false)
    # Compute households' utility - with normalisation if necessary 
    if normalise == true
        return (consumption ^ (1 - hhpar.rra) - 1)/(1 - hhpar.rra) - hhpar.phi * (labor ^ (1 + hhpar.frisch) - 1)/(1 + hhpar.frisch)
    else
        return (consumption ^ (1 - hhpar.rra))/(1 - hhpar.rra) - hhpar.phi * (labor ^ (1 + hhpar.frisch))/(1 + hhpar.frisch)
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
function get_opt_labor_from_FOC(c, rho, w, taxes, hhpar; neg_consumption_error = false)
    # num = (taxes.lambda_y * (1 - taxes.tau_y) * (rho * w)^(1 - taxes.tau_y)) * c^(-hhpar.rra) 
    # den = (hhpar.phi * (2 - taxes.lambda_c * (1 - taxes.tau_c) * c ^ (-taxes.tau_c)))
    if neg_consumption_error && c < 0 
        throw("Passed negative consumption to labor FOC!")
    else
        l_star = (((taxes.lambda_y * (1 - taxes.tau_y) * (rho * w)^(1 - taxes.tau_y)) * c^(-hhpar.rra)) / #Numerator
        (hhpar.phi * (2 - taxes.lambda_c * (1 - taxes.tau_c) * c ^ (-taxes.tau_c))) #Denominator - TBM This can be negative with negative tau_c!
        ) ^ (1 / (hhpar.frisch + taxes.tau_y)) # Exponent
    end
    return l_star
end

function get_opt_labor_with_zero_consumption(rho, a, a_prime, w, taxes; neg_labor_warning = true)
    # Compute optimal labor in case of zero consumption 
    # Derived from λ_y * (ρwℓ)^(1 - τ_y) + (1 + (1 - τ_k)r)a - a' = 0
    l_star = ((1 / (rho * w)) * ((a_prime - (1 + net_r) * a) / taxes.lambda_y) ^ (1 / (1 - taxes.tau_y)))

    # Check if negative labor is implied
    if neg_labor_warning && l_star < 0
        @warn ("Negative labor implied even with zero consumption! Household cannot pay interests on debt!")
    end
    return l_star
end

function get_opt_c_with_max_labor(rho, a, a_prime, w, net_r, taxes; max_labor = 1)
    # Compute optimal labor in case of zero consumption 
    # Derived solving c + T_c(c) =  λ_y * (ρw*max_labor)^(1 - τ_y) + (1 + (1 - τ_k)r)a - a'
    rhs = taxes.lambda_y * (rho * w * max_labor) ^ (1 - taxes.tau_y) + (1 + net_r) * a - a_prime

    # Solve for consumption and return
    return find_c_feldstein(rhs, taxes.lambda_c, taxes.tau_c; notax_upper=nothing)
end



