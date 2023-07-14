struct ChainConfiguration
    ω_k::Vector{Float64}
    η_k::Vector{Float64}
    β_jk::Vector{Vector{Float64}}
    Ω_j::Vector{Float64}
end

function get_closure(config::ChainConfiguration, ion_pair::Tuple{Int64,Int64}, μ::Float64, t::Float64)
    closure = 0
    for j in ion_pair
        for k in 1:length(config.ω_k)
            scale = config.Ω_j[j] * config.η_k[k] * config.β_jk[j][k]
            integral = im / (config.ω_k[k] - μ) * (exp(-im * t * config.ω_k[k] - μ) - 1)
            closure = closure + scale * integral
        end
    end
    return closure
end

function get_scaling_for_angle(config::ChainConfiguration, ion_pair::Tuple{Int64,Int64}, t::Float64, Θ::Float64)
    raw_angle = 0
    for k in 1:length(config.ω_k)
        mode_detune = config.ω_k[k] - detune
        scalar = (-1.0 * config.β_jk[ion_pair[0][k]] * config.β_jk[ion_pair[0][k]]
                  * config.Ω_j[ion_pair[0]] * config.Ω_j[ion_pair[1]] / 2 * mode_detune)
        integral = t + 1 / (mode_detune) * sin(mode_detune * t)
        raw_angle = raw_angle + scalar * integral
    end
    return raw_angle / Θ
end

chain = ChainConfiguration(
    [2.08321498e6, 2.176e6],
    [0.1, 0.1],
    [[1.0, 1.0], [-1.0, 1.0]],
    [1 / 2 / 300e-6, 1 / 2 / 300e-6],
)

freq = (2.08321498e6 + 2.176e6)/2
get_closure(chain, (1,2), freq, 1.0)