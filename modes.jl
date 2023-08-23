include("ion_spacing.jl")
using PlotlyJS
using LinearAlgebra
using Plots

function get_distance_matrix(positions::Vector{Float64})
    # returns 1/abs(x_i - x_j)
    n = length(positions)
    xii = []
    for pos in positions
        xii = vcat(xii, fill(pos, 1, n))
    end 
    abs_distance = 1 ./ broadcast(abs, xii - xii')
    replace!(abs_distance, Inf=>0)
    return abs_distance
end 

function get_axial_modes(trap::TrapVoltage)
    # Return axial mode in Hz and participation vectors
    positions = get_ion_spacing_for_voltage(trap)
    # off-diagonal terms: 2/abs(x_i - x_j)^3
    hessian = get_distance_matrix(positions)
    hessian = -2 * ELECTRON_CHARGE^2 / (4π * PERMITTIVITY) * (hessian .^3)
    # diagonal terms: second derivate of static potential + suming over 2/abs(x_i - x_j)^3 for all j
    function get_static_derivative(x0::Float64)
        # returns the second derivative of the static axial potential
        x0′ = x0 / EURIQA_x0
        # U = q * V
        return ELECTRON_CHARGE * EURIQA_v0 / EURIQA_x0^2 * (trap.x2 + trap.x3 * x0′ + trap.x4 * x0′^2 / 2.0) 
    end 
    coulomb_derivative = sum(hessian, dims=1)
    for i in 1:trap.num_ion
        hessian[i,i] = get_static_derivative(positions[i]) - coulomb_derivative[i]
    end 
    results = eigen(hessian) 
    if any(x->x<0, results.values)
        print("Hessian matrix should not have negative eigenvalues")
        return
    end 
    return sqrt.(results.values/ YB171_MASS) / 2π, results.vectors
end 

function get_radial_modes(trap::TrapVoltage, radial_com::Float64)
    # Radial_com: frequency of radial com mode in Hz
    # Return all radial mode in Hz, and participation vectors
    positions = get_ion_spacing_for_voltage(trap)
    # off-diagonal terms: 1/abs(x_i - x_j)^3
    hessian = get_distance_matrix(positions)
    hessian = 1 * ELECTRON_CHARGE^2 / (4π * PERMITTIVITY) * (hessian .^3)
    # diagonal terms: second derivative of radial confinement at y_0 + 1/abs(x_i - x_j)^3 summing over all j
    coulomb_derivative = sum(hessian, dims=1)
    for i in 1:trap.num_ion
        hessian[i,i] = YB171_MASS * (radial_com * 2π)^2 - coulomb_derivative[i]
    end 
    results = eigen(hessian) # ω2 is mω^2 
    if any(x->x<0, results.values)
        print("Hessian matrix should not have negative eigenvalues")
        return
    end 
    return sqrt.(results.values / YB171_MASS) / 2π, results.vectors
end

num_ion = 10
voltage = find_voltage_for_spacing(4.7e-6, num_ion, true, 0)
# modes, participation = get_axial_modes(voltage)
modes, participation = get_radial_modes(voltage, 2.369e6)
bar(modes,fill(1,voltage.num_ion, 1))
histogram(modes, bins=200, label="mode frequency")
xlabel!("Mode frequency (Hz)")
plot(heatmap(z=participation))