import gymnasium as gym
import numpy as np
from scipy.linalg import solve_continuous_are

# Physical parameters (MuJoCo default):
m = 0.1    # pole mass (kg)
M = 1.0    # cart mass (kg)
l = 0.5    # half pole length (m)
g = 9.8    # gravity (m/s^2)

# Continuous-time linearized state-space (around upright: theta = 0)
# State ordering for LQR: [x, x_dot, theta, theta_dot]
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m * g) / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m) * g / (l * M), 0]
])
B = np.array([[0], [1 / M], [0], [-1 / (l * M)]])

# LQR cost matrices: Q penalizes state error, R penalizes control effort
Q = np.diag([10.0, 1.0, 10.0, 1.0])  # state weights
R = np.array([[0.001]])               # control weight

# Solve continuous-time Algebraic Riccati equation
X = solve_continuous_are(A, B, Q, R)
# Compute LQR gain K = R^-1 B^T X
K = np.linalg.inv(R) @ (B.T @ X)
print("LQR gain K:", K)

# Create Mujoco environment (action space: Box(-3, 3, (1,)); obs: qpos (2), qvel (2))
env = gym.make("InvertedPendulum-v5", render_mode="human")
obs, _ = env.reset()

# Convert raw observation [qpos0, qpos1, qvel0, qvel1] to LQR state [x, x_dot, theta, theta_dot]
def extract_state(obs):
    x = obs[0]
    theta = obs[1]
    x_dot = obs[2]
    theta_dot = obs[3]
    return np.array([x, x_dot, theta, theta_dot])

# Run one episode
terminated = False
truncated = False
while not (terminated or truncated):
    state = extract_state(obs)
    # LQR control law: u = -K x
    u = -K.dot(state)
    # Clip to environment limits [-3, 3]
    u = np.clip(u, env.action_space.low, env.action_space.high)
    obs, reward, terminated, truncated, info = env.step(u)
    env.render()

env.close()