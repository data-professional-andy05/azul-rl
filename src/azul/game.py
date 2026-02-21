import numpy as np
import random
from .constants import (
    GRID_SIZE, 
    EMPTY,
    BLUE, YELLOW, RED, BLACK, WHITE,
    FIRST_PLAYER_TOKEN,
    PLAYABLE_COLORS,
    TILES_PER_COLOR,
    TILES_PER_FACTORY,
    FACTORY_COUNTS,
    FLOOR_LINE_CAPACITY
)
from .board import PlayerBoard

class AzulGame:
    def __init__(self, num_players=2):
        self.num_players = num_players
        if num_players not in FACTORY_COUNTS:
            raise ValueError(f"Invalid number of players: {num_players}")
        
        self.num_factories = FACTORY_COUNTS[num_players]
        self.players = [PlayerBoard() for _ in range(num_players)]
        
        self.bag = {} 
        self.box = {} 
        self.factories = np.zeros((self.num_factories, 6), dtype=np.int8)
        self.center = np.zeros(6, dtype=np.int8)
        
        self.first_player_token_available = True
        self.current_start_player = 0 
        self.current_player_idx = 0   
        self.round_number = 0
        
        # Stores debug logs for the last round
        self.round_logs = {}
        
        self.reset()

    def reset(self):
        for p in self.players:
            p.reset()
            
        self.bag = {c: TILES_PER_COLOR for c in PLAYABLE_COLORS}
        self.box = {c: 0 for c in PLAYABLE_COLORS}
        
        self.round_number = 0
        self.current_start_player = random.randint(0, self.num_players - 1)
        self.start_new_round()
        
        return self.get_global_state()

    def start_new_round(self):
        self.round_number += 1
        self.current_player_idx = self.current_start_player
        self.factories.fill(0)
        self.center.fill(0)
        self.first_player_token_available = True
        self.round_logs = {} # Clear logs
        
        for f_idx in range(self.num_factories):
            for _ in range(TILES_PER_FACTORY):
                color = self._draw_tile()
                if color is not None:
                    self.factories[f_idx, color] += 1

    def _draw_tile(self):
        total_bag = sum(self.bag.values())
        if total_bag == 0:
            total_box = sum(self.box.values())
            if total_box == 0: return None 
            self.bag = self.box.copy()
            self.box = {c: 0 for c in PLAYABLE_COLORS}
            total_bag = total_box

        choices = list(self.bag.keys())
        weights = list(self.bag.values())
        color = random.choices(choices, weights=weights, k=1)[0]
        self.bag[color] -= 1
        return color

    def step(self, action):
        source_idx, color, target_row = action
        player = self.players[self.current_player_idx]
        
        tiles_taken = 0
        
        if source_idx == -1: 
            if self.center[color] == 0:
                raise ValueError("Move Invalid: Color not in center.")
            tiles_taken = self.center[color]
            self.center[color] = 0
            if self.first_player_token_available:
                self.first_player_token_available = False
                self.current_start_player = self.current_player_idx
                player.add_tiles(-1, FIRST_PLAYER_TOKEN, 1) 
        else:
            if self.factories[source_idx, color] == 0:
                 raise ValueError("Move Invalid: Color not in factory.")
            tiles_taken = self.factories[source_idx, color]
            self.factories[source_idx, color] = 0
            for c in PLAYABLE_COLORS:
                remainder = self.factories[source_idx, c]
                if remainder > 0:
                    self.center[c] += remainder
                    self.factories[source_idx, c] = 0

        success = player.add_tiles(target_row, color, tiles_taken)
        if not success:
            player.add_tiles(-1, color, tiles_taken)

        if self._is_round_empty():
            self._end_round_processing()
            if not self.is_game_over():
                self.start_new_round()
        else:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        return self.get_global_state()

    def _is_round_empty(self):
        factories_empty = np.sum(self.factories) == 0
        center_empty = np.sum(self.center) == 0
        return factories_empty and center_empty

    def _end_round_processing(self):
        self.round_logs = {} 
        for i, p in enumerate(self.players):
            # Capture Verbose Logs
            _, discarded_tiles, logs = p.calculate_round_bonuses(verbose=True)
            self.round_logs[i] = logs
            
            for tile in discarded_tiles:
                if tile in self.box: self.box[tile] += 1

    def apply_end_game_bonuses(self):
        for p in self.players:
            p.calculate_end_game_score()

    def is_game_over(self):
        for p in self.players:
            for row in range(GRID_SIZE):
                if np.count_nonzero(p.wall[row]) == GRID_SIZE:
                    return True
        return False

    def get_global_state(self):
        return {
            "factories": self.factories.copy(),
            "center": self.center.copy(),
            "players": [p.get_state_vector() for p in self.players],
            "current_player": self.current_player_idx,
            "first_player_token": 1 if self.first_player_token_available else 0
        }