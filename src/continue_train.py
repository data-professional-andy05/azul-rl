import sys
import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rl_env import AzulEnv

# --- CONFIG ---
LOAD_MODEL_PATH = "models/ppo_azul_big_20M/azul_20M_final.zip" 
NEW_TOTAL_TIMESTEPS = 100000 
MODELS_DIR = "models/ppo_azul_big_20M_test"
LOGS_DIR = "logs"

def mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()

def make_env():
    env = AzulEnv(num_players=2) 
    env = Monitor(env) 
    env = ActionMasker(env, mask_fn) 
    return env 

def continue_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Recreate Environment
    env = DummyVecEnv([make_env])
    
    # 2. Load the Existing Brain
    print(f"Loading model from: {LOAD_MODEL_PATH}")
    # We don't need to specify policy_kwargs or architecture here, 
    # because .load() reads them from the zip file automatically!
    model = MaskablePPO.load(LOAD_MODEL_PATH, env=env, tensorboard_log=LOGS_DIR)

    print("\n" + "="*40)
    print("RESUMING TRAINING")
    print(f"Adding {NEW_TOTAL_TIMESTEPS} steps...")
    print("="*40 + "\n")
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=MODELS_DIR,
        name_prefix="azul_20M"
    )

    # 4. Train
    # reset_num_timesteps=False keeps the TensorBoard line continuous 
    # instead of starting back at 0 on the X-axis.
    model.learn(
        total_timesteps=NEW_TOTAL_TIMESTEPS, 
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False 
    )
    
    model.save(f"{MODELS_DIR}/azul_20M_finaltest")
    print("20 Million Steps Reached! Model saved.")

if __name__ == "__main__":
    continue_training()