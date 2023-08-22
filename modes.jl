include("ion_spacing.jl")
using LinearAlgebra

function get_distance_matrix(positions::Vector{Float64})
    # returns 1/abs(x_i - x_j)
    n = length(positions)
    xii = []
    for pos in positions
        xii = vcat(xii, fill(pos, 1, n))
    end 
    abs_distance = 1 ./ broadcast(abs, xii - xii')
    replace!(abs_distance, Inf=>0)
    @show abs_distance
    return abs_distance
end 

function get_axial_hessian(trap::TrapVoltage)
    positions = get_ion_spacing_for_voltage(trap)
    # off-diagonal terms: 2/abs(x_i - x_j)^3
    hessian = get_distance_matrix(positions)
    hessian = -2 * ELECTRON_CHARGE^2 / (4π * PERMITTIVITY) * (hessian .^3)
    @show hessian
    # diagonal terms: second derivate of static potential + suming over 2/abs(x_i - x_j)^3 for all j
    function get_static_derivative(x0::Float64)
        # returns the second derivative of the static axial potential
        x0′ = x0 / EURIQA_x0
        # something is not right here
        return EURIQA_v0 / EURIQA_x0^2 * (trap.x2 + trap.x3 * x0′ + trap.x4 * x0′^2 / 2.0) 
    end 
    coulomb_derivative = sum(hessian, dims=1)
    for i in 1:trap.num_ion
        hessian[i,i] = get_static_derivative(positions[i]) - coulomb_derivative[i]
    end 
    @show hessian
    return hessian
end 


function get_axial_modes(trap::TrapVoltage)
    hessian = get_axial_hessian(trap)
    ω2 = eigvals(hessian) # ω2 is mω^2 
    if any(x->x<0, ω2)
        print("Hessian matrix should not have negative eigenvalues")
        return
    end 
    return sqrt.(ω2 / YB171_MASS) / 2π
end


voltage = find_voltage_for_spacing(4.7e-6, 2, true, 0)
get_axial_modes(voltage)