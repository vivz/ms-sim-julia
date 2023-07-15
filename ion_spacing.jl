using Optim
using Plots

const EURIQA_x0 = 2.7408e-6
const EURIQA_v0 = 525.39e-6
const ELECTRON_CHARGE = 1.60217663e-19
const PERMITTIVITY= 8.8541878128e-12

struct TrapVoltage
    x1::Float64
    x2::Float64
    x3::Float64
    x4::Float64
    num_ion::Int64
end

# Calculates ion poosition given a voltage configuration

function get_voltage(trap::TrapVoltage, x::Float64)
    x′ = x / EURIQA_x0
    return EURIQA_v0 * (trap.x1 * x′ + trap.x2 * x′^2 / 2.0 + trap.x3 * x′^3 / 6.0 + trap.x4 * x′^4 / 24.0)
end

function get_coulomb_potential_at_x(ion_positions::Vector{Float64}, ion_index::Int64)
    # Calculate coulomb potential at one ion's position from other ions 
    potential = 0.0
    x = ion_positions[ion_index]
    for i in eachindex(ion_positions)
        if i != ion_index
            potential = potential + ELECTRON_CHARGE / (4π * PERMITTIVITY * abs(ion_positions[i] - x))
        end
    end
    return potential
end

function get_total_potential(trap::TrapVoltage, ion_positions::Vector{Float64})
    total_potential = 0.0
    for i in eachindex(ion_positions)
        trap_potential = get_voltage(trap, ion_positions[i])
        coulomb_potential = get_coulomb_potential_at_x(ion_positions, i)
        # divide by two because we counted twice
        total_potential = total_potential + trap_potential + coulomb_potential / 2.0
    end
    return total_potential
end

function calculate_gradient!(G, trap::TrapVoltage, ion_positions::Vector{Float64})
    # Calculate total potential's gradient wrt each ion position 
    for i in eachindex(G)
        x = ion_positions[i] / EURIQA_x0
        trap_gradient = EURIQA_v0 / EURIQA_x0 * (trap.x1 + trap.x2 * x + trap.x3 * x^2 / 2.0 + trap.x4 * x^3 / 6.0) 
        coulomb_gradient = 0.0
        for j in eachindex(ion_positions)
            if j != i
                distance = ion_positions[j] - ion_positions[i]
                coulomb_gradient = coulomb_gradient + sign(distance) / distance^2
            end 
        end 
        G[i] = trap_gradient + ELECTRON_CHARGE / (4π * PERMITTIVITY) * coulomb_gradient
    end 
end 

function get_ion_spacing_for_voltage(trap::TrapVoltage)
    # returns in um
    num_ion = trap.num_ion
    initial_ion_spacing = collect(LinRange(-num_ion / 2, num_ion / 2, num_ion)) * 1e-6
    res = optimize(
            params -> get_total_potential(trap, params), 
            (G, params) -> calculate_gradient!(G, trap, params), 
            initial_ion_spacing,
            Optim.Options(iterations = 50000)
        )
    print(res)
    return Optim.minimizer(res)
end

function get_deviation(ion_positions::Vector{Float64})
    num_ion = length(ion_positions)
    ion_spacings = Array{Float64}(undef, num_ion) 
    for i in 2:num_ion
        ion_spacings[i] = ion_positions[i] - ion_positions[i-1]
    end
    return std(ion_spacings)
end 

trap = TrapVoltage(0, 0.398, 0, 0.02, 31)
spacing = get_ion_spacing_for_voltage(trap)
scatter(spacing, zeros(trap.num_ion), label="data", minorgrid=true)
print(spacing)