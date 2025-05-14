import gymnasium as gym
import numpy as np
import time
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Physical parameters (MuJoCo default):
m = 0.1    # pole mass (kg)
M = 1.0    # cart mass (kg)
l = 0.5    # half pole length (m)
g = 9.8    # gravity (m/s^2)
I = 1 / 12 * m * l * l
b = 0.1
p = I * (M + m) + M * m * l * l

# Continuous-time linearized state-space (around upright: theta = 0)
# State ordering for LQR: [x, x_dot, theta, theta_dot]

A = np.array([[0, 1, 0, 0], [0, -((I + m * l * l) * b) / p, (m ** 2 * g * l ** 2) / p, 0], [0, 0, 0, 1],
                  [0, (-m * l * b) / p, (m * g * l * (M + m)) / p, 0]])
B = np.array([[0], [(I + m * l ** 2) / p], [0], [-m * l / p]])


# Callback function to track optimization progress
class OptimizationTracker:
    def __init__(self):
        self.costs = []
        self.params = []

    def __call__(self, xk):
        cost = simulate_episode(xk)
        self.costs.append(cost)
        self.params.append(xk.copy())
        return False


def simulate_episode(params, render=False, max_steps=1000):
    """Simulate with Q and R parameters"""
    Q = np.diag(np.exp(params[:4]))  # Q diagonal (exp ensures positive)
    R = np.diag(np.exp(params[4:]))  # R diagonal

    try:
        X = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ X)
    except:
        return np.inf  # Return high cost if Riccati fails

    env = gym.make("InvertedPendulum-v5", render_mode="human" if render else None)
    obs, _ = env.reset()

    total_cost = 0
    for step in range(max_steps):
        state = np.array([obs[0], obs[2], obs[1], obs[3]])  # [x, x_dot, theta, theta_dot]
        u = -K @ state
        u = np.clip(u, -3, 3)

        obs, _, terminated, truncated, _ = env.step(u)
        cost = state.T @ Q @ state + u.T @ R @ u
        total_cost += cost

        if terminated or truncated:
            break
        if render:
            env.render()
            time.sleep(0.02)

    env.close()
    return total_cost / (step + 1)  # Average cost per step


def plot_opt_comparison(tracker_lbfgs, tracker_nm, save=False):
    fig = plt.figure(figsize=(12, 6))

    # Cost vs iteration
    plt.subplot(1, 2, 1)
    plt.plot(tracker_lbfgs.costs, 'b-', label='L-BFGS-B')
    plt.plot(tracker_nm.costs, 'r-', label='Nelder-Mead')
    plt.xlabel('Iteration')
    plt.ylabel('Avergae Cost')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True)

    # Final cost comparison
    plt.subplot(1, 2, 2)
    methods = ['L-BFGS-B', 'Nelder-Mead']
    final_costs = [tracker_lbfgs.costs[-1], tracker_nm.costs[-1]]
    plt.bar(methods, final_costs, color=['blue', 'red'])
    plt.ylabel('Final Average Cost')
    plt.title('Final Performance Comparison')

    plt.tight_layout()
    if save:
        plt.savefig('optimization_comparison.png')
    return fig


def plot_param_evolution(tracker_lbfgs, tracker_nm, save=False):
    params_lbfgs = np.array(tracker_lbfgs.params)
    params_nm = np.array(tracker_nm.params)

    param_names = ['Q_x', 'Q_xdot', 'Q_theta', 'Q_thetador', 'R']

    fig = plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(3, 2, i+1)
        plt.plot(np.exp(params_lbfgs[:, i]), 'b-', label='L-BFGS-B')
        plt.plot(np.exp(params_nm[:, i]), 'r-', label='Nelder-Mead')
        plt.xlabel('Iteration')
        plt.ylabel(f"Parameter Value ({param_names[i]})")
        plt.title(f'{param_names[i]} Evolution')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig('parameter_evolution.png')
    return fig


def plot_cost_distribution(result_lbfgs, result_nm, save=False):
    costs_lbfgs = [simulate_episode(result_lbfgs.x) for _ in range(20)]
    costs_nm = [simulate_episode(result_nm.x) for _ in range(20)]

    fig = plt.figure(figsize=(10, 6))
    plt.boxplot([costs_lbfgs, costs_nm], labels=['L-BFGS-B', 'Nelder-Mead'])
    plt.ylabel('Average Cost per Episode')
    plt.title('Cost Distribution Across 20 Episodes')
    plt.grid(True)
    if save:
        plt.savefig('cost_distribution.png')
    return fig


# Optimization setup
initial_params = np.log([10., 1., 10., 1., 0.1])  # [Q_diag, R]
bounds = Bounds(np.log([0.1, 0.1, 0.1, 0.1, 0.001]),
                np.log([100., 100., 100., 100., 1.]))

# Create trackers for each method
tracker_lbfgs = OptimizationTracker()
tracker_nm = OptimizationTracker()

# L-BFGS-B Optimization
print("Running L-BFGS-B optimization...")
result_lbfgs = minimize(
    lambda p: simulate_episode(p),
    initial_params,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 500, 'disp': True},
    callback=tracker_lbfgs
)

# Nelder-Mead Optimization (no bounds support)
print("\nRunning Nelder-Mead optimization...")
result_nm = minimize(
    lambda p: simulate_episode(np.clip(p, bounds.lb, bounds.ub)),  # Manual bounds
    initial_params,
    method='Nelder-Mead',
    options={'maxiter': 200, 'adaptive': True, 'disp': True},
    callback=tracker_nm
)

# Print optimization statistics
print("\n Optimization Statistics:")
print(f"L-BFGS-B: {len(tracker_lbfgs.costs)} iterations, final cost: {tracker_lbfgs.costs[-1]:.2f}")
print(f"Nelder-Mead: {len(tracker_nm.costs)} iterations, final cost: {tracker_nm.costs[-1]:.2f}")

# Compare results
results = {
    'L-BFGS-B': (result_lbfgs.x, result_lbfgs.fun),
    'Nelder-Mead': (result_nm.x, result_nm.fun)
}
best_method = min(results, key=lambda k: results[k][1])
best_params = results[best_method][0]

# Convert back from log space
optimized_params = np.exp(best_params)
Q_opt = np.diag(optimized_params[:4])
R_opt = np.diag(optimized_params[4:])

print(f"\nBest method: {best_method}")
print(f"Optimized Q: {np.diag(Q_opt)}")
print(f"Optimized R: {np.diag(R_opt)}")
print(f"Average cost: {results[best_method][1]:.2f}")

# Generate plots
fig1 = plot_opt_comparison(tracker_lbfgs, tracker_nm, save=True)
fig2 = plot_param_evolution(tracker_lbfgs, tracker_nm, save=True)
fig3 = plot_cost_distribution(result_lbfgs, result_nm, save=True)

plt.show()

# Final simulation with best parameters
print("\nRunning final simulation...")
simulate_episode(best_params, render=True, max_steps=2000)
