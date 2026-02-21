import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.azul.game import AzulGame
from src.azul.constants import (
    GRID_SIZE, PLAYABLE_COLORS, FACTORY_COUNTS, 
    TILES_PER_FACTORY, ID_TO_COLOR
)

class AzulEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, num_players=2, render_mode=None):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode
        self.game = AzulGame(num_players)
        self.action_space = spaces.Discrete(180)
        dummy_obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=0, high=100, shape=dummy_obs.shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action_idx):
        source_idx, color_idx, dest_idx = self.decode_action(action_idx)
        color_id = PLAYABLE_COLORS[color_idx]
        game_source = -1 if source_idx == self.game.num_factories else source_idx
        game_dest = -1 if dest_idx == 5 else dest_idx
        
        prev_scores = [p.score for p in self.game.players]
        
        try:
            self.game.step((game_source, color_id, game_dest))
        except ValueError:
            return self._get_obs(), -100, True, False, {"valid": False}
            
        terminated = self.game.is_game_over()
        truncated = False
        
        if terminated:
            self.game.apply_end_game_bonuses()
            
        current_scores = [p.score for p in self.game.players]
        total_delta = sum(current_scores) - sum(prev_scores)
        
        reward = total_delta
        reward -= 0.1
        
        return self._get_obs(), reward, terminated, truncated, {"valid": True}

    def action_masks(self):
        mask = np.zeros(180, dtype=bool)
        player = self.game.players[self.game.current_player_idx]
        for action_idx in range(180):
            source_idx, color_idx, dest_idx = self.decode_action(action_idx)
            color_id = PLAYABLE_COLORS[color_idx]
            game_source = -1 if source_idx == self.game.num_factories else source_idx
            game_dest = -1 if dest_idx == 5 else dest_idx
            
            valid_source = False
            if game_source == -1: 
                if self.game.center[color_id] > 0: valid_source = True
            else: 
                if game_source < self.game.num_factories and self.game.factories[game_source, color_id] > 0:
                    valid_source = True
            
            if not valid_source: continue 
            if game_dest == -1: 
                mask[action_idx] = True
            else:
                if player.can_add_to_pattern_line(game_dest, color_id):
                    mask[action_idx] = True
        return mask

    def _get_obs(self):
        state = self.game.get_global_state()
        obs = np.concatenate([
            state['factories'].flatten(),
            state['center'].flatten(),
            np.concatenate([p for p in state['players']]),
            np.eye(self.num_players)[state['current_player']],
            [state['first_player_token']]
        ])
        return obs.astype(np.float32)

    def decode_action(self, action_idx):
        dest = action_idx % 6
        remaining = action_idx // 6
        color = remaining % 5
        source = remaining // 5
        return source, color, dest

    def render(self):
        pass # Not used by play_vs_ai (uses custom print)