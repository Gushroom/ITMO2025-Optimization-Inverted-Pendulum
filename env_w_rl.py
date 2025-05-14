import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


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


# Convert raw observation [qpos0, qpos1, qvel0, qvel1] to LQR state [x, x_dot, theta, theta_dot]
def extract_state(obs):
    x = obs[0]
    theta = obs[1]
    x_dot = obs[2]
    theta_dot = obs[3]
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)


# Not really working
class QROptimizer(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1)
        )

    def forward(self, state):
        out = self.net(state)
        Q_diag = torch.exp(out[:4])
        R = torch.exp(out[:, -1:])
        return Q_diag, R

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.float32).reshape(-1, 1)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_rl_lqr(env, episodes=1000, batch_size=64, gamma=0.99):
    state_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qr_optimizer = QROptimizer(state_dim).to(device)
    target_net = QROptimizer(state_dim).to(device)
    target_net.load_state_dict(qr_optimizer.state_dict())

    optimizer = optim.Adam(qr_optimizer.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10000)

    for episode in range(episodes):
        obs, _ = env.reset()
        state = extract_state(obs)
        print("Initial extracted state shape:", state.shape)
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            Q_diag, R = qr_optimizer(state_tensor)

            Q = torch.diag_embed(Q_diag).squeeze(0).detach().cpu().numpy()
            R = R.squeeze(0).detach().cpu().numpy().reshape(1,1)

            try:
                X = solve_continuous_are(A, B, Q, R)
                K = np.linalg.inv(R) @ (B.T @ X)
                action = (-K @ state).clip(-3, 3)
            except:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = extract_state(next_obs)
            done = terminated or truncated

            assert state.shape == (4,), f"Bad state shape in push: {state.shape}"
            assert next_state.shape == (4,), f"Bad next_state shape in push: {next_state.shape}"
            buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states_t = torch.FloatTensor(states).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                rewards_t = torch.FloatTensor(rewards).to(device)
                dones_t = torch.FloatTensor(dones).to(device)

                with torch.no_grad():
                    next_Q_diag, next_R = target_net(next_states_t)
                    next_Q = torch.diag_embed(next_Q_diag)  # Shape: [batch_size, 4, 4]
                    next_V = torch.bmm(
                        torch.bmm(next_states_t.unsqueeze(1), next_Q),
                        next_states_t.unsqueeze(2)
                    ).squeeze()
                    targets = rewards_t + gamma * (1 - dones_t) * next_V

                Q_diag, R = qr_optimizer(states_t)
                Q = torch.diag_embed(Q_diag)  # Shape: [batch_size, 4, 4]
                current_V = torch.bmm(
                    torch.bmm(states_t.unsqueeze(1), Q),
                    states_t.unsqueeze(2)
                ).squeeze()

                loss = nn.MSELoss()(current_V, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if done:
                break
            state = next_state

        if episode % 10 == 0:
            target_net.load_state_dict(qr_optimizer.state_dict())

        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    return qr_optimizer

def evaluate_el_lqr(qr_optimizer, env, render=True):
    state_history = []
    Q_history = []
    R_history = []

    obs, _ = env.reset()
    state = extract_state(obs)
    total_reward = 0

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        Q_diag, R = qr_optimizer(state_tensor)

        Q = torch.diag_embed(Q_diag).squeeze(0).detach().cpu().numpy()
        R = R.squeeze(0).detach().cpu().numpy()

        try:
            X = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ (B.T @ X)
            action = (-K @ state).clip(-3, 3)
        except:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(obs)

        state_history.append(state)
        Q_history.append(np.diag(Q))
        R_history.append(R[0,0])
        total_reward += reward

        if render:
            env.render()
            time.sleep(0.02)

        if terminated or truncated:
            break

        state = next_state

    env.close()
    return np.array(state_history), np.array(Q_history), np.array(R_history), total_reward


def plot_results(state_history, Q_history, R_history, save=False):
    fig = plt.figure(figsize=(15, 10))

    state_names = ['x', 'x_dot', 'theta', 'theta_dot']

    for i in range(4):
        plt.subplot(3, 2, i + 1)
        plt.plot(state_history[:, i], label=state_names[i])
        plt.xlabel('Time Step')
        plt.ylabel(state_names[i])
        plt.title(f'{state_names[i]} Evolution')
        plt.grid(True)
        plt.legend()

        # Q matrix evolution
    plt.subplot(3, 2, 5)
    for i in range(4):
        plt.plot(Q_history[:, i], label=f'Q_{i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Q Value')
    plt.title('Q Matrix Diagonal Elements')
    plt.grid(True)
    plt.legend()

    # R evolution
    plt.subplot(3, 2, 6)
    plt.plot(R_history, label='R', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('R Value')
    plt.title('Control Weight Evolution')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig('rl_lqr_results.png')
    return fig


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("\nStarting training...")
        qr_optimizer = train_rl_lqr(env, episodes=500)

        print("\nEvaluating trained controller...")
        state_history, Q_history, R_history, reward = evaluate_el_lqr(qr_optimizer, env)
        plot_results(state_history, Q_history, R_history)
        plt.show()

        print(f"\nFinal Evaluation Reward: {reward:.2f}")

    finally:
        env.close()
