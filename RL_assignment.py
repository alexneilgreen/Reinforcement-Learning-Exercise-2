import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

# Force-generate a 1200-step sinusoidal + noise speed profile
speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
df_fake = pd.DataFrame({"speed": speeds})
df_fake.to_csv(CSV_FILE, index=False)
print(f"Created {CSV_FILE} with {DATA_LEN} steps.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# Create lead vehicle position data based on lead vehicle speed
# Lead vehicle will follow the reference speed profile with some variations
lead_speeds = full_speed_data + 1 * np.sin(0.05 * np.arange(DATA_LEN)) + 0.5 * np.random.randn(DATA_LEN)
lead_positions = np.cumsum(lead_speeds)  # Integrate speed to get position

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, positions, chunk_size):
    """
    Splits `data` and corresponding `positions` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    position_episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        position_chunk = positions[start:end]
        episodes.append(chunk)
        position_episodes.append(position_chunk)
        start = end
    return episodes, position_episodes

# ------------------------------------------------------------------------
# Define constants for ACC
# ------------------------------------------------------------------------
MIN_SAFE_DISTANCE = 5.0  # Minimum safe following distance in meters
MAX_SAFE_DISTANCE = 30.0  # Maximum desired following distance in meters
MAX_ACCEL = 2.0  # Maximum acceleration in m/s^2
MAX_DECEL = -2.0  # Maximum deceleration in m/s^2

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Adaptive Cruise Control (ACC) training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-2,2] m/s^2
      - observation: [current_speed, reference_speed, distance_to_lead_vehicle]
      - reward: weighted combination of speed error, distance error, and jerk penalty
    """

    def __init__(self, episodes_list, position_episodes_list, delta_t=1.0):
        super().__init__()
        self.episodes_list = episodes_list
        self.position_episodes_list = position_episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        # Constrained acceleration range for realistic vehicle dynamics
        self.action_space = spaces.Box(low=MAX_DECEL, high=MAX_ACCEL, shape=(1,), dtype=np.float32)
        
        # Extended observation space to include distance to lead vehicle
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([50.0, 50.0, 100.0]), 
            dtype=np.float32
        )

        # Episode-specific
        self.current_episode = None
        self.current_position_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0
        self.ego_position = 0.0
        self.prev_action = 0.0  # Store previous action for jerk calculation
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.current_position_episode = self.position_episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]
        self.ego_position = self.current_position_episode[self.step_idx] - 20.0  # Start at a reasonable distance
        self.prev_action = 0.0

        # Calculate current distance to lead vehicle
        lead_position = self.current_position_episode[self.step_idx]
        distance_to_lead = lead_position - self.ego_position

        obs = np.array([self.current_speed, self.ref_speed, distance_to_lead], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # Apply realistic acceleration limits
        accel = np.clip(action[0], MAX_DECEL, MAX_ACCEL)
        
        # Calculate jerk (change in acceleration)
        jerk = (accel - self.prev_action) / self.delta_t
        self.prev_action = accel
        
        # Update ego vehicle state
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0
        
        # Update position based on speed
        self.ego_position += self.current_speed * self.delta_t
        
        # Get reference speed and lead vehicle position
        ref_speed = self.full_data[self.idx]
        lead_position = self.lead_positions[self.idx]
        
        # Calculate distance to lead vehicle
        distance_to_lead = lead_position - self.ego_position
        
        # Calculate lead vehicle speed
        lead_speed = 0
        if self.idx > 0:
            lead_speed = lead_position - self.lead_positions[self.idx-1]
        
        # Speed error
        speed_error = abs(self.current_speed - ref_speed)
        
        # Distance error
        if distance_to_lead < MIN_SAFE_DISTANCE:
            # High penalty for unsafe distance
            distance_error = 5.0 * (MIN_SAFE_DISTANCE - distance_to_lead)
        elif distance_to_lead > MAX_SAFE_DISTANCE:
            # Moderate penalty for exceeding max distance
            distance_error = 1.0 * (distance_to_lead - MAX_SAFE_DISTANCE)
        else:
            # No penalty within safe range
            distance_error = 0.0
        
        # Comprehensive ACC reward
        # Balance between speed following, safe distance, and comfort
        reward = -speed_error - 2.0 * distance_error - 0.1 * abs(jerk) - 0.05 * abs(accel)

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed, distance_to_lead], dtype=np.float32)
        info = {
            "speed_error": speed_error, 
            "distance_error": distance_error,
            "distance": distance_to_lead,
            "jerk": jerk,
            "lead_speed": lead_speed
        }
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Adaptive Cruise Control (ACC) testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed, distance_to_lead_vehicle]
      - reward: weighted combination of speed error, distance error, and jerk penalty
    """

    def __init__(self, full_data, lead_positions, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.lead_positions = lead_positions
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        # Actions, Observations with realistic constraints
        self.action_space = spaces.Box(low=MAX_DECEL, high=MAX_ACCEL, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([50.0, 50.0, 100.0]), 
            dtype=np.float32
        )

        self.idx = 0
        self.current_speed = 0.0
        self.ego_position = 0.0
        self.prev_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        self.ego_position = self.lead_positions[0] - 20.0  # Start at a reasonable distance
        self.prev_action = 0.0
        
        ref_speed = self.full_data[self.idx]
        distance_to_lead = self.lead_positions[self.idx] - self.ego_position
        
        obs = np.array([self.current_speed, ref_speed, distance_to_lead], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # Apply realistic acceleration limits
        accel = np.clip(action[0], MAX_DECEL, MAX_ACCEL)
        
        # Calculate jerk (change in acceleration)
        jerk = (accel - self.prev_action) / self.delta_t
        self.prev_action = accel
        
        # Update ego vehicle state
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0
        
        # Update position based on speed
        self.ego_position += self.current_speed * self.delta_t
        
        # Get reference speed and lead vehicle position
        ref_speed = self.full_data[self.idx]
        lead_position = self.lead_positions[self.idx]
        
        # Calculate distance to lead vehicle
        distance_to_lead = lead_position - self.ego_position
        
        # Speed error
        speed_error = abs(self.current_speed - ref_speed)
        
        # Distance error
        if distance_to_lead < MIN_SAFE_DISTANCE:
            # High penalty for unsafe distance
            distance_error = 5.0 * (MIN_SAFE_DISTANCE - distance_to_lead)
        elif distance_to_lead > MAX_SAFE_DISTANCE:
            # Moderate penalty for exceeding max distance
            distance_error = 1.0 * (distance_to_lead - MAX_SAFE_DISTANCE)
        else:
            # No penalty within safe range
            distance_error = 0.0
        
        # Comprehensive ACC reward
        # Balance between speed following, safe distance, and comfort
        reward = -speed_error - 2.0 * distance_error - 0.1 * abs(jerk) - 0.05 * abs(accel)

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed, distance_to_lead], dtype=np.float32)
        info = {
            "speed_error": speed_error, 
            "distance_error": distance_error,
            "distance": distance_to_lead,
            "jerk": jerk,
            "lead_speed": (lead_position - self.lead_positions[self.idx-2]) / self.delta_t if (self.idx-1) > 0 else 0
        }
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (with additional metrics)
# ------------------------------------------------------------------------
class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, total_timesteps, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        self.episode_speed_errors = []
        self.episode_distance_errors = []
        self.episode_jerks = []
        self.total_timesteps = total_timesteps
        self.last_progress = -1  # For tracking progress percentage
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward', 'average_speed_error', 'average_distance_error', 'average_jerk'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        info = self.locals.get('infos', [{}])[-1]
        
        # Calculate progress percentage
        progress_percent = int((t / self.total_timesteps) * 100)
        
        # Only update display when percentage changes to avoid flooding console
        if progress_percent > self.last_progress:
            self.last_progress = progress_percent
            progress_bar = f"[{'=' * (progress_percent // 2)}>{' ' * (50 - (progress_percent // 2))}]"
            print(f"\rTraining: {progress_bar} {progress_percent}% ({t}/{self.total_timesteps} timesteps)", end="")
        
        speed_error = info.get('speed_error', 0)
        distance_error = info.get('distance_error', 0)
        jerk = info.get('jerk', 0)
        
        self.episode_rewards.append(reward)
        self.episode_speed_errors.append(speed_error)
        self.episode_distance_errors.append(distance_error)
        self.episode_jerks.append(jerk)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            avg_speed_error = np.mean(self.episode_speed_errors)
            avg_distance_error = np.mean(self.episode_distance_errors)
            avg_jerk = np.mean(np.abs(self.episode_jerks))
            
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward, avg_speed_error, avg_distance_error, avg_jerk])
            
            self.logger.record("reward/average_reward", avg_reward)
            self.logger.record("metrics/average_speed_error", avg_speed_error)
            self.logger.record("metrics/average_distance_error", avg_distance_error)
            self.logger.record("metrics/average_jerk", avg_jerk)
            
            self.episode_rewards.clear()
            self.episode_speed_errors.clear()
            self.episode_distance_errors.clear()
            self.episode_jerks.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_acc_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    # Add model selection argument
    parser.add_argument(
        "--model",
        type=str,
        default="SAC",
        choices=["SAC", "PPO", "TD3", "DDPG"],
        help="RL algorithm to use (SAC, PPO, TD3, DDPG)."
    )
    # Add hyperparameter arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training."
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=200000,
        help="Replay buffer size."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor."
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=-1.0,
        help="Entropy coefficient. Use -1.0 for 'auto'."
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="256,256",
        help="Network architecture (comma-separated list of layer sizes)."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1200,
        help="Total timesteps for training."
    )
    
    args = parser.parse_args()

    log_dir = args.output_dir
    
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    chunk_size = args.chunk_size
    print(f"[INFO] Using chunk_size = {chunk_size}")
    print(f"[INFO] Selected model: {args.model}")
    print(f"[INFO] ACC implementation active with safe distance range: {MIN_SAFE_DISTANCE}-{MAX_SAFE_DISTANCE}m")

    # Parse net_arch from string to list of integers
    net_arch = [int(size) for size in args.net_arch.split(",")]
    print(f"[INFO] Network architecture: {net_arch}")

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list, position_episodes_list = chunk_into_episodes(full_speed_data, lead_positions, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, position_episodes_list, delta_t=1.0)

    train_env = DummyVecEnv([make_train_env])

    # 5C) Build the model based on selected algorithm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=net_arch, activation_fn=nn.ReLU)
    
    # Handle ent_coef special case
    ent_coef = 'auto' if args.ent_coef == -1.0 else args.ent_coef
    
    # Common parameters for all algorithms
    common_params = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "device": device,
        "gamma": args.gamma,
    }
    
    # Initialize the selected algorithm with appropriate parameters
    if args.model == "SAC":
        model = SAC(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            ent_coef=ent_coef
        )
    elif args.model == "PPO":
        model = PPO(
            **common_params,
            batch_size=args.batch_size,
            n_steps=2048,  # PPO-specific parameter
            ent_coef=0.01 if args.ent_coef == -1.0 else args.ent_coef
        )
    elif args.model == "TD3":
        model = TD3(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau
        )
    elif args.model == "DDPG":
        model = DDPG(
            **common_params,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.set_logger(logger)

    total_timesteps = args.total_timesteps
    callback = CustomLoggingCallback(log_dir, total_timesteps)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"{args.model.lower()}_acc_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, lead_positions, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    distances = []
    rewards = []
    actions = []
    jerks = []
    lead_speeds = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        predicted_speeds.append(obs[0])     # current_speed
        reference_speeds.append(obs[1])     # reference_speed
        distances.append(obs[2])            # distance to lead vehicle
        rewards.append(reward)
        jerks.append(info["jerk"])
        lead_speeds.append(info["lead_speed"])
        
        if terminated or truncated:
            break

    # Calculate metrics
    avg_test_reward = np.mean(rewards)
    avg_speed_error = np.mean([abs(p - r) for p, r in zip(predicted_speeds, reference_speeds)])
    avg_distance = np.mean(distances)
    distance_in_range = np.mean([(d >= MIN_SAFE_DISTANCE and d <= MAX_SAFE_DISTANCE) for d in distances]) * 100

    # Calculate jerk metrics
    avg_jerk = np.mean(np.abs(jerks))
    jerk_variance = np.var(jerks)
    
    # Calculate quantitative metrics
    speed_errors = np.array([abs(p - r) for p, r in zip(predicted_speeds, reference_speeds)])
    squared_speed_errors = speed_errors**2
    
    # Distance errors (distance outside the safe range)
    distance_errors = []
    for d in distances:
        if d < MIN_SAFE_DISTANCE:
            distance_errors.append(MIN_SAFE_DISTANCE - d)
        elif d > MAX_SAFE_DISTANCE:
            distance_errors.append(d - MAX_SAFE_DISTANCE)
        else:
            distance_errors.append(0)
    distance_errors = np.array(distance_errors)
    
    # Speed difference with lead vehicle
    speed_diffs = np.array([abs(p - l) for p, l in zip(predicted_speeds, lead_speeds)])
    
    # Mean metrics
    mae_speed = np.mean(speed_errors)
    mse_speed = np.mean(squared_speed_errors)
    rmse_speed = np.sqrt(mse_speed)
    
    mae_distance = np.mean(distance_errors)
    mean_speed_diff = np.mean(speed_diffs)

    # Add metrics to test results dataframe
    results_path = os.path.join(log_dir, f"{args.model.lower()}_acc_results_chunk{chunk_size}.csv")
    results_df = pd.DataFrame({
        "timestep": range(len(predicted_speeds)),
        "reference_speed": reference_speeds,
        "predicted_speed": predicted_speeds,
        "lead_speed": lead_speeds,
        "distance": distances,
        "speed_error": speed_errors,
        "distance_error": distance_errors,
        "speed_diff": speed_diffs,
        "reward": rewards,
        "action": actions,
        "jerk": jerks
    })
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Test results saved to: {results_path}")

    # Save metrics to separate file
    metrics_path = os.path.join(log_dir, f"{args.model.lower()}_acc_metrics_chunk{chunk_size}.csv")
    metrics_df = pd.DataFrame({
        'metric': [
            'MAE_Speed', 'MSE_Speed', 'RMSE_Speed', 
            'MAE_Distance', 'Mean_Speed_Diff',
            'Mean_Absolute_Jerk', 'Jerk_Variance',
            'Distance_In_Range_Percent'
        ],
        'value': [
            mae_speed, mse_speed, rmse_speed, 
            mae_distance, mean_speed_diff,
            avg_jerk, jerk_variance,
            distance_in_range
        ]
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    # Create only the required plots for ACC metrics
    # 1. Speed tracking plot (reference vs predicted vs lead)
    plt.figure(figsize=(12, 6))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Ego Vehicle Speed", linestyle="-")
    plt.plot(lead_speeds, label="Lead Vehicle Speed", linestyle=":", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Speed Tracking Performance ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    speed_plot_path = os.path.join(log_dir, f"1_acc_speed_tracking_plot.png")
    plt.savefig(speed_plot_path)
    plt.close()

    # 2. Distance to lead vehicle plot
    plt.figure(figsize=(12, 6))
    plt.plot(distances, label="Following Distance", color="blue")
    plt.axhline(y=MIN_SAFE_DISTANCE, color='red', linestyle='--', label=f"Min Safe Distance ({MIN_SAFE_DISTANCE}m)")
    plt.axhline(y=MAX_SAFE_DISTANCE, color='orange', linestyle='--', label=f"Max Desired Distance ({MAX_SAFE_DISTANCE}m)")
    plt.fill_between(range(len(distances)), MIN_SAFE_DISTANCE, MAX_SAFE_DISTANCE, alpha=0.2, color='green', label="Safe Range")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title(f"Following Distance to Lead Vehicle ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    distance_plot_path = os.path.join(log_dir, f"2_acc_distance_plot.png")
    plt.savefig(distance_plot_path)
    plt.close()

    # 3. Jerk plot for ride comfort assessment
    plt.figure(figsize=(12, 6))
    plt.plot(jerks, label="Jerk", color="purple")
    plt.axhline(y=avg_jerk, color='black', linestyle='--', label=f"Mean Abs Jerk = {avg_jerk:.4f} m/s³")
    plt.xlabel("Timestep")
    plt.ylabel("Jerk (m/s³)")
    plt.title(f"Ride Comfort: Jerk Analysis ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    jerk_plot_path = os.path.join(log_dir, f"3_acc_jerk_plot.png")
    plt.savefig(jerk_plot_path)
    plt.close()

    # 4. Speed difference with lead vehicle
    plt.figure(figsize=(12, 6))
    plt.plot(speed_diffs, label="Speed Difference", color="green")
    plt.axhline(y=mean_speed_diff, color='black', linestyle='--', label=f"Mean = {mean_speed_diff:.4f} m/s")
    plt.xlabel("Timestep")
    plt.ylabel("Speed Difference (m/s)")
    plt.title(f"Speed Difference: Ego vs Lead Vehicle ({args.model}, chunk_size={chunk_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    speed_diff_plot_path = os.path.join(log_dir, f"4_acc_speed_diff_plot.png")
    plt.savefig(speed_diff_plot_path)
    plt.close()

    # Print summary of hyperparameters and metrics for ACC
    print("\n--- ACC Training Configuration and Results Summary ---")
    print(f"Algorithm: {args.model}")
    print(f"Chunk size: {chunk_size}")
    print(f"Network architecture: {net_arch}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    if args.model in ["SAC", "TD3", "DDPG"]:
        print(f"Buffer size: {args.buffer_size}")
        print(f"Tau: {args.tau}")
    if args.model == "SAC":
        print(f"Entropy coefficient: {ent_coef}")
    print(f"Gamma: {args.gamma}")
    print(f"Total timesteps: {total_timesteps}")
    
    print("\n--- ACC Performance Metrics ---")
    print(f"Speed tracking: MAE = {mae_speed:.4f}, RMSE = {rmse_speed:.4f}")
    print(f"Time with distance in safe range: {distance_in_range:.1f}%")
    print(f"Mean distance to lead vehicle: {avg_distance:.2f}m")
    print(f"Mean distance error outside safe range: {mae_distance:.4f}m")
    print(f"Mean absolute jerk: {avg_jerk:.4f} m/s³ (lower is better for comfort)")
    print(f"Jerk variance: {jerk_variance:.4f} m²/s⁶")
    print(f"Mean speed difference with lead vehicle: {mean_speed_diff:.4f} m/s")
    print("-----------------------------------")


if __name__ == "__main__":
    main()