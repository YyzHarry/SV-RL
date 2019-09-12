module MountainCar

using MDPs, PGFPlots
using DelimitedFiles

export state_space, action_space, transition, reward, simulate,
       viz_policy, viz_trajectory
export XMIN, XMAX, VMIN, VMAX


# mdp
const MX = 50
const MV = 50
const N = 1000

# state
const XMIN = -1.2
const XMAX = 0.5
const VMIN = -0.07
const VMAX = 0.07

# action
const AMIN = -1.0
const AMAX = 1.0

# reward
const TERM_REWARD = 10
const NONTERM_REWARD = -1

# test
const T = 500


function state_space(mx::Int64=MX, mv::Int64=MV)
    xs = range(XMIN, stop=XMAX, length=mx)
    vs = range(VMIN, stop=VMAX, length=mv)
    return RectangleGrid(xs, vs)
end


function action_space(n::Int64=N)
    return range(AMIN, stop=AMAX, length=n)
end


function transition(state::Vector{Float64}, action::Float64)
    snext = zeros(length(state))
    snext[2] = state[2] + 0.001 * action - 0.0025 * cos(3 * state[1])
    snext[2] = clip(snext[2], VMIN, VMAX)
    snext[1] = clip(state[1] + snext[2], XMIN, XMAX)
    return snext
end


function clip(val::Float64, minval::Float64, maxval::Float64)
    return min(maxval, max(minval, val))
end


function reward(state::Vector{Float64}, action::Float64)
    if state[1] == XMAX
        return TERM_REWARD
    else
        return NONTERM_REWARD
    end
end


function simulate(mdp::MDP, policy::Policy, state::Vector{Float64})
    trajectory = zeros(T, dimensions(mdp.S))
    actions = zeros(T)
    for t = 1:T
        trajectory[t, :] = state
        if state[1] == XMAX
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
    function getmap(x::Float64, v::Float64)
        return policy!(policy, get_belief(mdp, [x, v]))
    end

    p = Axis([Plots.Image(getmap, (XMIN, XMAX), (VMIN, VMAX),
                      xbins = 250, ybins = 250,
                      colormap = ColorMaps.Named("Blues"))],
          width="14cm", height="14cm",
          xlabel="position", ylabel="speed",
          title=title)
    p
    if saveplot
      PGFPlots.save(name, p)
    end
end


function viz_trajectory(trajectory::Matrix{Float64}, actions::Vector{Float64},
                        title1::String="position over time", title2::String="acceleration over time",
                        saveplot::Bool=false, name::String="")
    g = GroupPlot(2, 1, groupStyle="horizontal sep=1.5cm")
    push!(g, Axis([Plots.Linear(trajectory[:, 1])],
                  width="12cm", height="12cm",
                  xlabel="time", ylabel="position",
                  title=title1))
    push!(g, Axis([Plots.Linear(actions)],
                  width="12cm", height="12cm",
                  xlabel="time", ylabel="acceleration",
                  title=title2))
    g
    if saveplot
      PGFPlots.save(name, g)
    end
end


end