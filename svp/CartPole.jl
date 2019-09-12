module CartPole

using MDPs
using DelimitedFiles
using PGFPlots

export state_space, action_space, transition, reward, simulate,
       viz_policy, viz_trajectory
export THETAMIN, THETAMAX, THETADOTMIN, THETADOTMAX, XMIN, XMAX, XDOTMIN, XDOTMAX, T


# mdp
const MTHETA = 10
const MTHETADOT = 10
const MX = 10
const MXDOT = 10
const N = 1000

# state
const THETAMIN = float(-pi / 2)
const THETAMAX = float(pi / 2)
const THETADOTMIN = -3.0
const THETADOTMAX = 3.0
const XMIN = -2.4
const XMAX = 2.4
const XDOTMIN = -3.5
const XDOTMAX = 3.5

# action
const AMIN = -10
const AMAX = 10

# transition
const GRAVITY = 9.8
const MASSCART = 1.0
const MASSPOLE = 0.1
const LENGTH = 0.5
# const FORCE_MAG = 10
const DT = 0.1  # time intervals

# reward
const K = 15

# test
const T = 200

# visualization
const EPS = 1e-4


function state_space(mtheta::Int64=MTHETA, mthetadot::Int64=MTHETADOT, mx::Int64=MX, mxdot::Int64=MXDOT)
    theta = range(THETAMIN, stop=THETAMAX, length=mtheta)
    thetadot = range(THETADOTMIN, stop=THETADOTMAX, length=mthetadot)
    x = range(XMIN, stop=XMAX, length=mx)
    xdot = range(XDOTMIN, stop=XDOTMAX, length=mxdot)
    return RectangleGrid(theta, thetadot, x, xdot)
end


function action_space(n::Int64=N)
    return range(AMIN, stop=AMAX, length=n)
end


function wrap_around(angle::Float64)
    while angle < -pi / 2
        angle = -pi / 2
    end
    while angle > pi / 2
        angle = pi / 2
    end
    return angle
end


function clip(val::Float64, minval::Float64, maxval::Float64)
    return min(maxval, max(minval, val))
end


function transition(state::Vector{Float64}, action::Float64)
    temp = (action + MASSPOLE * LENGTH * state[2] * state[2] * sin(state[1])) / (MASSPOLE + MASSCART)
    theta_acc = (GRAVITY * sin(state[1]) - cos(state[1]) * temp) / (LENGTH * (4/3 - MASSPOLE * cos(state[1]) * cos(state[1]) / (MASSPOLE + MASSCART)))
    x_acc = temp - MASSPOLE * LENGTH * theta_acc * cos(state[1]) / (MASSPOLE + MASSCART)
    
    snext = zeros(length(state))
    snext[1] = wrap_around(state[1] + DT * state[2])
    snext[2] = clip(state[2] + DT * theta_acc, THETADOTMIN, THETADOTMAX)
    snext[3] = clip(state[3] + DT * state[4], XMIN, XMAX)
    snext[4] = clip(state[4] + DT * x_acc, XDOTMIN, XDOTMAX)
    return snext
end


function reward(state::Vector{Float64}, action::Float64)
    return (cos(K * state[1]))^4
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
        state = mdp.transition(state, action)
        actions[t] = action
    end
    return trajectory, actions
end


function viz_policy(mdp::MDP, policy::Policy, title::String="Policy heatmap", saveplot::Bool=false, name::String="")
    function getmap(p::Float64, v::Float64)
        return policy!(policy, get_belief(mdp, [p, v]))
    end

    p = Axis([Plots.Image(getmap, (THETAMIN + EPS, THETAMAX - EPS), 
                      (THETADOTMIN + EPS, THETADOTMAX - EPS),
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