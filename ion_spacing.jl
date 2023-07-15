using Optim
using Plots
using LineSearches

const EURIQA_x0 = 2.7408e-6
const EURIQA_v0 = 525.39e-6
const ELECTRON_CHARGE = 1.60217663e-19
const PERMITTIVITY = 8.8541878128e-12

struct TrapVoltage
    x1::Float64
    x2::Float64
    x3::Float64
    x4::Float64
    num_ion::Int64
end

# Calculates ion poosition given a voltage configuration

function get_voltage(trap::TrapVoltage, x::Float64)
    # Calculate voltage in the trap electrode
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
    # For a given voltage configuration and ion positions, calculate total potential
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
    # Calculate total potential's analytical gradient wrt each ion position 
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
    # Find ion positions with minimum energy for a given voltage 
    num_ion = trap.num_ion
    initial_ion_spacing = collect(LinRange(-num_ion / 2, num_ion / 2, num_ion)) * 1e-6
    res = optimize(
        params -> get_total_potential(trap, params),
        (G, params) -> calculate_gradient!(G, trap, params),
        initial_ion_spacing,
        Optim.Options(iterations=50000)
    )
    return Optim.minimizer(res)
end

function get_deviation(ion_positions::Vector{Float64}, spacing::Float64, num_edge_ion::Int64=0)
    # Calculates deviation from the ideal equal-spaced configuration
    num_ion = length(ion_positions) - num_edge_ion * 2
    ideal_positions = collect(LinRange(-num_ion * spacing / 4, num_ion * spacing / 4, num_ion))
    return sum((ion_positions[num_edge_ion+1:end-num_edge_ion] - ideal_positions) .^ 2) * 1e10
end

function get_voltage(spacing::Float64, num_ion::Int64, quadratic::Bool=true, num_edge_ion::Int64=0)
    initial_params = quadratic ? [0.1] : [0.2, 0.001] # only using x2 and x4 for optimization
    lower = quadratic ? [0] : [0, 0]
    upper = quadratic ? [Inf] : [Inf, Inf]
    inner_optimizer = GradientDescent(linesearch=LineSearches.BackTracking(order=3))
    res = Optim.optimize(
        params -> get_deviation(
            get_ion_spacing_for_voltage(TrapVoltage(0.0, params[1], 0.0, quadratic ? 0.0 : params[2], num_ion)), spacing, num_edge_ion),
        lower,
        upper,
        initial_params,
        Fminbox(inner_optimizer)
    )
    # calculate best voltage for a given ion initial_ion_spacing
    print(res)
    voltages = Optim.minimizer(res)
    return TrapVoltage(0.0, voltages[1], 0.0, quadratic ? 0.0 : voltages[2], num_ion)
end 

# trap = TrapVoltage(0, 0.4, 0, 0.0, 2)
ideal_voltage = get_voltage(4.7e-6, 2, true, 0)
print(ideal_voltage)
pos = get_ion_spacing_for_voltage(ideal_voltage)
scatter(pos, zeros(trap.num_ion), minorgrid=true)
ylims!(-1,1)



