module MDPs

using GridInterpolations
using SparseArrays
using Printf
using DelimitedFiles
using PyCall

export MDP, Policy, policy!, value_iteration, get_belief, dimensions, RectangleGrid, __init__

function __init__()
    py"""
    import numpy as np
    from scipy.stats import bernoulli
    from fancyimpute import SoftImpute, BiScaler, NuclearNormMinimization

    def generate_mask(mask_prob=0.5):
        # generate mask for limited observation
        # change NUM_STATES for different planning tasks
        NUM_STATES = 10000
        NUM_ACTIONS = 1000
        RESIZE_SHAPE = (NUM_STATES, NUM_ACTIONS)
        mask = bernoulli.rvs(p=mask_prob, size=(RESIZE_SHAPE[0], RESIZE_SHAPE[1])).astype(float)
        mask[mask < 1] = np.nan
        
        return mask
    
    def bregman(image, mask, weight, eps=1e-3, max_iter=100):
        rows, cols, dims = image.shape
        rows2 = rows + 2
        cols2 = cols + 2
        total = rows * cols * dims
        shape_ext = (rows2, cols2, dims)

        u = np.zeros(shape_ext)
        dx = np.zeros(shape_ext)
        dy = np.zeros(shape_ext)
        bx = np.zeros(shape_ext)
        by = np.zeros(shape_ext)

        u[1:-1, 1:-1] = image
        u[0, 1:-1] = image[1, :]
        u[1:-1, 0] = image[:, 1]
        u[-1, 1:-1] = image[-2, :]
        u[1:-1, -1] = image[:, -2]

        i = 0
        rmse = np.inf
        lam = 2 * weight
        norm = (weight + 4 * lam)

        while i < max_iter and rmse > eps:
            rmse = 0
            for k in range(dims):
                for r in range(1, rows + 1):
                    for c in range(1, cols + 1):
                        uprev = u[r, c, k]
                        ux = u[r, c + 1, k] - uprev
                        uy = u[r + 1, c, k] - uprev
                        if mask[r - 1, c - 1]:
                            unew = (lam * (u[r + 1, c, k] + u[r - 1, c, k] +
                                           u[r, c + 1, k] + u[r, c - 1, k] +
                                           dx[r, c - 1, k] - dx[r, c, k] +
                                           dy[r - 1, c, k] - dy[r, c, k] -
                                           bx[r, c - 1, k] + bx[r, c, k] -
                                           by[r - 1, c, k] + by[r, c, k]
                                           ) + weight * image[r - 1, c - 1, k]
                                    ) / norm
                        else:
                            unew = (u[r + 1, c, k] + u[r - 1, c, k] +
                                    u[r, c + 1, k] + u[r, c - 1, k] +
                                    dx[r, c - 1, k] - dx[r, c, k] +
                                    dy[r - 1, c, k] - dy[r, c, k] -
                                    bx[r, c - 1, k] + bx[r, c, k] -
                                    by[r - 1, c, k] + by[r, c, k]
                                    ) / 4.0
                        u[r, c, k] = unew
                        rmse += (unew - uprev) ** 2
                        bxx = bx[r, c, k]
                        byy = by[r, c, k]

                        s = ux + bxx
                        if s > 1 / lam:
                            dxx = s - 1 / lam
                        elif s < -1 / lam:
                            dxx = s + 1 / lam
                        else:
                            dxx = 0
                        s = uy + byy
                        if s > 1 / lam:
                            dyy = s - 1 / lam
                        elif s < -1 / lam:
                            dyy = s + 1 / lam
                        else:
                            dyy = 0
                        dx[r, c, k] = dxx
                        dy[r, c, k] = dyy
                        bx[r, c, k] += ux - dxx
                        by[r, c, k] += uy - dyy

            rmse = np.sqrt(rmse / total)
            i += 1

        return np.squeeze(np.asarray(u[1:-1, 1:-1]))
    
    def tvm(input_array, keep_prob=0.8, lambda_tv=0.03):
        # expand input arr to 3-dims
        input_array = np.expand_dims(input_array, axis=2)
        mask = np.random.uniform(size=input_array.shape[:2])
        mask = mask < keep_prob
        return bregman(input_array, mask, weight=2.0/lambda_tv)
    
    def recover(mask, tmp_q_table, mask_prob=0.5):
        observed_q_table = mask * tmp_q_table
        recovered_q_table = SoftImpute(verbose=False).fit_transform(observed_q_table)
        
        return recovered_q_table
    """
end


const VTOL = 1e-6
const MAXITER = 200
const GAMMA = 0.95
const SAVELOC = "../data/qvalue.csv"


mutable struct MDP
    S           :: RectangleGrid
    nstate      :: Int64
    A           :: Vector{Float64}
    naction     :: Int64
    transition  :: Function
    reward      :: Function
    
    function MDP(S, A, transition, reward)
        return new(S, length(S), A, length(A), transition, reward)
    end
end


mutable struct Policy
    Q       :: Matrix{Float64}
    A       :: Vector{Float64}
    naction :: Int64
    qvals   :: Vector{Float64}
    
    function Policy(Q, A)
        return new(Q, A, length(A), zeros(length(A)))
    end
end


function policy!(policy::Policy, belief::SparseMatrixCSC{Float64, Int64})
    fill!(policy.qvals, 0.0)
    for iaction in 1:policy.naction
        for ib in 1:length(belief.rowval)
            policy.qvals[iaction] += 
                belief.nzval[ib] * policy.Q[belief.rowval[ib], iaction]
        end
    end
    ibest = argmax(policy.qvals)
    return policy.A[ibest]
end


function saveq(Q::Matrix{Float64}, saveloc::String)
    writedlm(saveloc, Q, ',')
end


function value_iteration(mdp::MDP, save::Bool=false, saveloc::String=SAVE_LOC, verbose::Bool=true)
    nstate = length(mdp.S)
    naction = length(mdp.A)
    V = zeros(nstate)
    Q = zeros(nstate, naction)
    state = zeros(dimensions(mdp.S))
    cputime = 0.0
    
    println("Starting value iteration...")
    iter = 0
    for iter = 1:MAXITER
        tic()
        residual = 0.0
        prob = 0.2
        mask = py"generate_mask"(prob)
        
        # structured value-based planning
        for istate = 1:nstate
            for iaction = 1:naction
                if !isnan(mask[istate, iaction])
                    ind2x!(mdp.S, istate, state)
                    action = mdp.A[iaction]
                    snext = mdp.transition(state, action)
                    vnext = interpolate(mdp.S, V, snext)
                    Qprev = Q[istate, iaction]
                    Q[istate, iaction] = mdp.reward(state, action) + GAMMA * vnext
                    residual += (Q[istate, iaction] - Qprev)^2
                end
            end
        end
        
        # reconstruct via matrix estimation
        Q = py"recover"(mask, Q, prob)
        
        V = [maximum(Q[istate, :]) for istate = 1:nstate]
        
        iter_time = toc()
        cputime += iter_time
        residual /= (nstate * naction * prob)
        
        if verbose
            @printf("Iteration %d: average residual = %.2e, cputime = %.2e sec\n", 
                    iter, residual, iter_time)
        end
        
        if residual < VTOL
            break
        end

        if iter == MAXITER
            println("Maximum number of iterations reached!")
        end
    end
    @printf("Value iteration took %d iterations and %.2e sec\n", iter, cputime)

    if save
        saveq(Q, saveloc)
    end

    return Policy(Q, mdp.A)
end


function get_belief(mdp::MDP, state::Vector{Float64})
    belief = spzeros(mdp.nstate, 1)
    indices, weights = interpolants(mdp.S, state)
    belief[indices] = weights
    return belief
end


end