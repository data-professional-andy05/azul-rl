import sys
import os
import torch as th
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rl_env import AzulEnv

# --- HYPERPARAMETERS ---
TOTAL_TIMESTEPS = 300_000   
LEARNING_RATE = 0.0003      
N_STEPS = 2048              
BATCH_SIZE = 64             
GAMMA = 0.99                

def mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()

def make_env():
    env = AzulEnv(num_players=2) 
    env = Monitor(env) 
    env = ActionMasker(env, mask_fn) 
    return env 

def train():
    # New folder for the Big Model
    models_dir = "models/ppo_azul_big" 
    logs_dir = "logs"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = DummyVecEnv([make_env])

    # --- BIG NETWORK ARCHITECTURE ---
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tensorboard_log=logs_dir,
        device="auto",
        policy_kwargs=policy_kwargs
    )

    print("\n" + "="*40)
    print("BIG BRAIN AGENT (256x256) INITIALIZED")
    print(model.policy)
    print("="*40 + "\n")

    print(f"Training started for {TOTAL_TIMESTEPS} timesteps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="azul_big"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save(f"{models_dir}/azul_big_final")
    print("Training Complete!")

if __name__ == "__main__":
    train()