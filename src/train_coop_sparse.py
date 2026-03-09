import sys
import os
import gymnasium as gym
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rl_env import AzulEnv

# --- CONFIGURATION ---
MODELS_DIR = "models/coop_sparse"
LOGS_DIR = "logs/coop_sparse"
TOTAL_TIMESTEPS = 5_000_000  # 5 Million Steps

# Save a model every 50,000 steps. 
# You will get: model_50000.zip, model_100000.zip, etc.
SAVE_FREQ = 50_000 

class CoopSparseAzulEnv(AzulEnv):
    def step(self, action_idx):
        # 1. Capture REAL Scores Before
        prev_scores = [p.score for p in self.game.players]
        
        # 2. Execute Move
        obs, _, terminated, truncated, info = super().step(action_idx)
        
        # 3. Capture REAL Scores After
        current_scores = [p.score for p in self.game.players]
        
        # 4. Cooperative Reward (Sum of Real Deltas)
        total_delta = sum(current_scores) - sum(prev_scores)
        
        reward = total_delta
        reward -= 0.1 # Step penalty
        
        return obs, reward, terminated, truncated, info

def mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()

def make_env():
    env = CoopSparseAzulEnv(num_players=2) 
    env = Monitor(env) 
    env = ActionMasker(env, mask_fn) 
    return env 

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    env = DummyVecEnv([make_env])

    # Big Brain Architecture
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=LOGS_DIR,
        device="auto",
        policy_kwargs=policy_kwargs
    )

    print(f"--- STARTING COOP SPARSE TRAINING (Target: {TOTAL_TIMESTEPS}) ---")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODELS_DIR,
        name_prefix="coop_sparse"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save(f"{MODELS_DIR}/coop_sparse_final")
    print("Done.")

if __name__ == "__main__":
    train()