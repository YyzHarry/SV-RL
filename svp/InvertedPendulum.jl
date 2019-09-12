module InvertedPendulum

using MDPs
using DelimitedFiles
using PGFPlots

export state_space, action_space, transition, reward, simulate,
       viz_policy, viz_trajectory
export PMIN, PMAX, VMIN, VMAX, T


# mdp
const MP = 50
const MV = 50
const N = 1000

# state
const PMIN = float(-pi)
const PMAX = float(pi)
const VMIN = -10.0
const VMAX = 10.0

# action
const AMIN = -1.0
const AMAX = 1.0

# transition
const A = 1.0
const B = 1.0
const DT = MV / MP * pi / VMAX  # time intervals
const NOISE = deg2rad(0.5)

# reward
const K = 1.0
const R = 0.1

# test
const T = 200

# visualization
const EPS = 1e-4


function state_space(mp::Int64=MP, mv::Int64=MV)
    p = range(PMIN, stop=PMAX, length=mp)
    v = range(VMIN, stop=VMAX, length=mv)
    return RectangleGrid(p, v)
end


function action_space(n::Int64=N)
    return range(AMIN, stop=AMAX, length=n)
end


function wrap_around(angle::Float64)
    while angle < -pi
        angle += 2 * pi
    end
    while angle > pi
        angle -= 2 * pi
    end
    return angle
end


function transition(state::Vector{Float64}, action::Float64, noise::Float64=0.0)
    snext = zeros(length(state))
    snext[1] = wrap_around(state[1] + DT * state[2])
    snext[2] = state[2] + DT * (A * sin(state[1]) - B * state[2] + action) + noise
    return snext
end


function reward(state::Vector{Float64}, action::Float64)
    return exp(K * (cos(state[1]) - 1)) - R * action^2 - 1
end


function simulate(mdp::MDP, policy::Policy, state::Vector{Float64})
    trajectory = zeros(T, dimensions(mdp.S))
    actions = zeros(T)
    for t = 1:T
        trajectory[t, :] = state
        if state[1] == 0.0
            trajectory = trajectory[1:t, :]
            actions = actions[1:t - 1]
            break
        end
        action = policy!(policy, get_belief(mdp, state))
        state = mdp.transition(state, action, NOISE * randn())
        actions[t] = action
    end
    return trajectory, actions
end


function viz_policy(mdp::MDP, policy::Policy, title::String="Policy heatmap", saveplot::Bool=false, name::String="")
    function getmap(p::Float64, v::Float64)
        return policy!(policy, get_belief(mdp, [p, v]))
    end

    p = Axis([Plots.Image(getmap, (PMIN + EPS, PMAX - EPS), 
                      (VMIN + EPS, VMAX - EPS),
                      xbins = 250, ybins = 250,
                      colormap = ColorMaps.Named("Blues"))],
          width="14cm", height="14cm",
          xlabel="angle", ylabel="angular speed",
          title=title)
    p
    if saveplot
      PGFPlots.save(name, p)
    end
end


function rad2deg_vec(trajectory::Vector{Float64})
    l = length(trajectory)
    out = zeros(l)
    for i = 1:l
        out[i] = rad2deg(trajectory[i])
    end
    return out
end


function viz_trajectory(trajectory::Matrix{Float64}, actions::Vector{Float64},
                        title1::String="angular position over time", title2::String="control input over time",
                        saveplot::Bool=false, name::String="")
    g = GroupPlot(2, 1, groupStyle="horizontal sep=1.5cm")
    push!(g, Axis([Plots.Linear(rad2deg_vec(trajectory[:, 1]))],
                  width="12cm", height="12cm",
                  xlabel="time", ylabel="angle (deg)",
                  title=title1))
    push!(g, Axis([Plots.Linear(actions)],
                  width="12cm", height="12cm",
                  xlabel="time", ylabel="input",
                  title=title2))
    g
    if saveplot
      PGFPlots.save(name, g)
    end
end


end