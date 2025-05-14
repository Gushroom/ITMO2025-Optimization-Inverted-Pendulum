import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import solve_continuous_are

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


state_history = []
control_history = []
time_history = []


# Convert raw observation [qpos0, qpos1, qvel0, qvel1] to LQR state [x, x_dot, theta, theta_dot]
def extract_state(obs):
    x = obs[0]
    theta = obs[1]
    x_dot = obs[2]
    theta_dot = obs[3]
    return np.array([x, x_dot, theta, theta_dot])


for step in range(1000):
    state = extract_state(obs)
    u = -K @ state
    u = np.clip(u, env.action_space.low, env.action_space.high)[0]

    obs, _, terminated, truncated, _ = env.step([u])
    env.render()

    time_history.append(step * 0.02)
    state_history.append(state)
    control_history.append(u)

    if terminated or truncated:
        break

env.close()

state_history = np.array(state_history)
control_history = np.array(control_history)
time_history = np.array(time_history)

def plot_state_evolution(state_history, control_history, time_history, save=False):
    state_names = ['x', 'x_dot', 'theta', 'theta_dot']

    fig = plt.figure(figsize=(15, 10))

    for i in range(4):
        plt.subplot(3, 2, i + 1)
        plt.plot(time_history, state_history[:, i], 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel(state_names[i])
        plt.title(f'{state_names[i]} Evolution')
        plt.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig('state_evolution.png')
    return fig

plot_state_evolution(state_history, control_history, time_history, save=True)
plt.show()

# # Run one episode
# terminated = False
# truncated = False
# max_steps = 1000
# step_count = 0
#
# while not (terminated or truncated):
#     state = extract_state(obs)
#     # LQR control law: u = -K x
#     u = -K.dot(state)
#     # Clip to environment limits [-3, 3]
#     u = np.clip(u, env.action_space.low, env.action_space.high)
#     obs, reward, terminated, truncated, info = env.step(u)
#     env.render()
#
#     time.sleep(0.02)
#     step_count += 1
#
# time.sleep(5)
#
# env.close()
