using Optim
using Plots

const euriqa_x0 = 2.7408e-6
const euriqa_v0 = 525.39e-6
const coulomb_constant = 8.99e9
const electron_charge = 1.60217663e-19
const permittivity= 8.8541878128e-12

struct TrapVoltage
    x1::Float64
    x2::Float64
    x3::Float64
    x4::Float64
    num_ion::Int64
end

function get_voltage(trap::TrapVoltage, x::Float64)
    x′ = x / euriqa_x0
    return euriqa_v0 * (trap.x1 * x′ + trap.x2 * x′^2 / 2.0 + trap.x3 * x′^3 / 6.0 + trap.x4 * x′^4 / 24.0)
end

function get_coulomb_potential_at_x(ion_positions::Vector{Float64}, x::Float64)
    potential = 0
    for ion_pos in ion_positions
        if ion_pos != x
            potential = potential + electron_charge / (8π * permittivity * abs(ion_pos - x))
        end
    end
    return potential
end

function get_total_potential(trap::TrapVoltage, ion_positions::Vector{Float64})
    total_potential = 0
    for i in 1:length(ion_positions)
        voltage_potential = get_voltage(trap, ion_positions[i])
        coulomb_potential = get_coulomb_potential_at_x(ion_positions, ion_positions[i])
        total_potential = total_potential + voltage_potential + coulomb_potential
    end
    return total_potential
end

function get_ion_spacing_for_voltage(trap::TrapVoltage)
    # returns in um
    num_ion = trap.num_ion
    initial_ion_spacing = collect(LinRange(-num_ion / 2, num_ion / 2, num_ion))
    res = optimize(params -> get_total_potential(trap, params), initial_ion_spacing)
    print(res)
    return Optim.minimizer(res) * 1e6
end
 
trap = TrapVoltage(0, 0.4, 0, 0, 3)
num_ion = trap.num_ion
initial_ion_spacing = collect(LinRange(-trap.num_ion / 2, trap.num_ion / 2, trap.num_ion))
spacing = get_ion_spacing_for_voltage(trap)
scatter(spacing, zeros(trap.num_ion), label="data", minorgrid=true)