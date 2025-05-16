import gymnasium as gym
import numpy as np
import time
from scipy.linalg import solve_continuous_are
from scipy.optimize import differential_evolution, dual_annealing
import matplotlib.pyplot as plt

# Physical parameters (MuJoCo default)
m = 0.1    # pole mass (kg)
M = 1.0    # cart mass (kg)
l = 0.5    # half pole length (m)
g = 9.8    # gravity (m/s^2)
I = 1 / 12 * m * l * l
b = 0.1
p = I * (M + m) + M * m * l * l

# Linearized system matrices
A = np.array([[0, 1, 0, 0], [0, -((I + m * l * l) * b) / p, (m ** 2 * g * l ** 2) / p, 0], [0, 0, 0, 1],
                  [0, (-m * l * b) / p, (m * g * l * (M + m)) / p, 0]])
B = np.array([[0], [(I + m * l ** 2) / p], [0], [-m * l / p]])

# Simulation function for an LQR controller

def simulate_episode(params, max_steps=1000):
    # Build Q, R
    Q = np.diag(np.exp(params[:4]))
    R = np.diag(np.exp(params[4:]))
    try:
        X = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ X)
    except Exception:
        return np.inf

    env = gym.make("InvertedPendulum-v5")
    obs, _ = env.reset()
    total_cost = 0.0
    for t in range(max_steps):
        state = np.array([obs[0], obs[2], obs[1], obs[3]])
        u = -K @ state
        u = np.clip(u, -3, 3)
        obs, _, done, trunc, _ = env.step(u)
        cost = state.T @ Q @ state + u.T @ R @ u
        total_cost += cost
        if done or trunc:
            break
    env.close()
    return total_cost / (t+1)

# Tracker classes
class DETracker:
    def __init__(self):
        self.costs = []
        self.params = []
    def __call__(self, xk, convergence=None):
        c = simulate_episode(xk)
        self.costs.append(c)
        self.params.append(xk.copy())
        return False

class SATracker:
    def __init__(self):
        self.costs = []
        self.params = []
    def __call__(self, xk, f, context):
        # xk is current solution, f is current cost
        self.costs.append(f)
        self.params.append(xk.copy())
        return False

initial_params = np.log([10., 1., 10., 1., 0.1])  # [Q_diag, R]
bounds = [(np.log(0.1), np.log(100)),  # Q_x
          (np.log(0.1), np.log(100)),  # Q_xdot
          (np.log(0.1), np.log(100)),  # Q_theta
          (np.log(0.1), np.log(100)),  # Q_thetador
          (np.log(0.001), np.log(1))]  # R

# Differential Evolution
tracker_de = DETracker()
print("=== Differential Evolution Optimization ===")
result_de = differential_evolution(
    func=lambda p: simulate_episode(p),
    x0=initial_params,
    bounds=bounds,
    strategy='best1bin',
    maxiter=50,
    popsize=15,
    tol=1e-3,
    callback=tracker_de,
    disp=True
)
print(f"DE best cost: {result_de.fun:.2f}")

# Simulated Annealing (Dual Annealing)
tracker_sa = SATracker()
print("\n=== Simulated Annealing Optimization ===")
result_sa = dual_annealing(
    func=lambda p: simulate_episode(p),
    x0=initial_params,
    bounds=bounds,
    maxiter=1000,
    callback=tracker_sa,
    initial_temp=5230.0,
    visit=2.62,
    accept=-5.0
)
print(f"SA best cost: {result_sa.fun:.2f}")

# Extract and display best configs
methods = ['DE', 'SA']
results = {'DE': result_de, 'SA': result_sa}
for m in methods:
    x = results[m].x
    c = results[m].fun
    vals = np.exp(x)
    print(f"\nMethod: {m}")
    print("  Cost:", f"{c:.2f}")
    print("  Q diag:", np.round(vals[:4], 3))
    print("  R:", np.round(vals[4:], 3))

# Plotting utilities
param_names = ['Q_x', 'Q_xdot', 'Q_theta', 'Q_thetador', 'R']

def plot_progress(trackers, labels):
    plt.figure(figsize=(10, 4))
    for tr, lab in zip(trackers, labels):
        plt.plot(tr.costs, label=lab)
    plt.xlabel('Iteration/Generation')
    plt.ylabel('Average Cost')
    plt.title('Optimization Progress Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_params(trackers, labels):
    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(3, 2, i+1)
        for tr, lab in zip(trackers, labels):
            arr = np.exp(np.array(tr.params))
            plt.plot(arr[:, i], marker='o', label=lab)
        plt.title(param_names[i])
        plt.xlabel('Iteration/Gen')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Display progress
plot_progress([tracker_de, tracker_sa], ['DE', 'SA'])
plot_params([tracker_de, tracker_sa], ['DE', 'SA'])

# Final cost distributions
def plot_distribution(results, labels, runs=20):
    data = []
    for m in labels:
        costs = [simulate_episode(results[m].x) for _ in range(runs)]
        data.append(costs)
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=labels)
    plt.title(f'Cost Distribution over {runs} runs')
    plt.ylabel('Average Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_distribution(results, ['DE', 'SA'])
