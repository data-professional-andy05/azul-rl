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
        # Validate player count based on rules
        if num_players not in FACTORY_COUNTS:
            raise ValueError(f"Invalid number of players: {num_players}. Must be 2, 3, or 4.")
        
        self.num_factories = FACTORY_COUNTS[num_players]
        self.players = [PlayerBoard() for _ in range(num_players)]
        
        # Game State Variables
        self.bag = {} # Current supply
        self.box = {} # Discard pile (Recycling)
        
        # Factory State: Matrix [Factory_ID, Color_ID] -> Count
        # We use 6 columns (0=Empty, 1-5=Colors). Row 0 is Factory 0.
        # This makes it easy for the Neural Network to read "How many Reds in Factory 1?"
        self.factories = np.zeros((self.num_factories, 6), dtype=np.int8)
        
        # Center State: Array [Color_ID] -> Count
        self.center = np.zeros(6, dtype=np.int8)
        
        self.first_player_token_available = True
        self.current_start_player = 0 # Who starts the round
        self.current_player_idx = 0   # Who moves now
        self.round_number = 0
        
        self.reset()

    def reset(self):
        """Resets the entire game state."""
        # Reset all boards
        for p in self.players:
            p.reset()
            
        # Refill Bag (20 of each color)
        self.bag = {c: TILES_PER_COLOR for c in PLAYABLE_COLORS}
        self.box = {c: 0 for c in PLAYABLE_COLORS}
        
        self.round_number = 0
        self.current_start_player = random.randint(0, self.num_players - 1)
        self.start_new_round()
        
        return self.get_global_state()

    def start_new_round(self):
        """Sets up the factories for a new round."""
        self.round_number += 1
        self.current_player_idx = self.current_start_player
        
        # Reset Factories and Center
        self.factories.fill(0)
        self.center.fill(0)
        
        # Place First Player Token in Center
        self.first_player_token_available = True
        
        # Fill Factories
        for f_idx in range(self.num_factories):
            for _ in range(TILES_PER_FACTORY):
                color = self._draw_tile()
                if color is not None:
                    self.factories[f_idx, color] += 1

    def _draw_tile(self):
        """Draws a random tile from the bag. Refills from box if empty."""
        total_bag = sum(self.bag.values())
        
        if total_bag == 0:
            # Bag empty, refill from box
            total_box = sum(self.box.values())
            if total_box == 0:
                return None # Truly out of tiles (rare)
            
            # Move box to bag
            self.bag = self.box.copy()
            self.box = {c: 0 for c in PLAYABLE_COLORS}
            total_bag = total_box

        # Weighted random draw
        choices = list(self.bag.keys())
        weights = list(self.bag.values())
        color = random.choices(choices, weights=weights, k=1)[0]
        
        self.bag[color] -= 1
        return color

    def step(self, action):
        """
        Executes a move.
        Action format: tuple (source_idx, color, target_row_idx)
        - source_idx: 0 to N-1 (Factories), -1 (Center)
        - color: 1 to 5
        - target_row_idx: 0 to 4 (Pattern Lines), -1 (Floor Line)
        
        Returns: (observation, reward, done, info)
        """
        source_idx, color, target_row = action
        player = self.players[self.current_player_idx]
        
        # 1. VALIDATION (Basic Logic)
        # Detailed move validation happens in the RL environment wrapper,
        # but the game engine must enforce physical reality.
        tiles_taken = 0
        penalty_tiles = 0
        
        # 2. EXECUTE PICK FROM SOURCE
        if source_idx == -1: 
            # Picking from CENTER
            if self.center[color] == 0:
                raise ValueError("Move Invalid: Color not in center.")
            
            tiles_taken = self.center[color]
            self.center[color] = 0
            
            # Handle First Player Token
            if self.first_player_token_available:
                self.first_player_token_available = False
                self.current_start_player = self.current_player_idx
                # Token goes to floor line (1 penalty)
                player.add_tiles(-1, FIRST_PLAYER_TOKEN, 1) # Internal ID for token
                
        else:
            # Picking from FACTORY
            if self.factories[source_idx, color] == 0:
                 raise ValueError("Move Invalid: Color not in factory.")
                 
            tiles_taken = self.factories[source_idx, color]
            self.factories[source_idx, color] = 0
            
            # Move remaining tiles in this factory to Center
            for c in PLAYABLE_COLORS:
                remainder = self.factories[source_idx, c]
                if remainder > 0:
                    self.center[c] += remainder
                    self.factories[source_idx, c] = 0

        # 3. EXECUTE PLACE ON BOARD
        # The board logic handles overflow to floor line automatically
        # Note: If move is strictly illegal (e.g. placing Red on a row that has Blue),
        # the Board returns False. Ideally, the RL Agent filters these out before calling step.
        success = player.add_tiles(target_row, color, tiles_taken)
        
        if not success:
            # If the specific row placement was illegal (e.g., wrong color),
            # standard rules say: "If you pick tiles you cannot place, they ALL fall to the floor."
            # We enforce this penalty here.
            player.add_tiles(-1, color, tiles_taken)

        # 4. CHECK END OF ROUND
        if self._is_round_empty():
            self._end_round_processing()
            if not self.is_game_over():
                self.start_new_round()
        else:
            # Pass turn to next player
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        return self.get_global_state()

    def _is_round_empty(self):
        """Check if all factories and center are empty."""
        factories_empty = np.sum(self.factories) == 0
        center_empty = np.sum(self.center) == 0
        return factories_empty and center_empty

    def _end_round_processing(self):
        """Trigger wall tiling and scoring for all players."""
        for p in self.players:
            p.calculate_round_bonuses()
            
            # Move tiles from floor line back to BOX (Recycling)
            # (In board.py, we just cleared the floor line, but we didn't track *what* colors were there.
            # To be perfectly accurate for counting cards, we should have returned the colors.
            # For simplicity in V1, we assume infinite supply or just ignore the specific colors discarded.)
            pass 

    def is_game_over(self):
        """Game ends if any player has completed a horizontal row."""
        for p in self.players:
            # Check rows (axis 1 sum is not enough, we need to check contiguous row completion)
            # Actually, standard rule: If a row has 5 tiles.
            for row in range(GRID_SIZE):
                if np.count_nonzero(p.wall[row]) == GRID_SIZE:
                    return True
        return False

    def get_global_state(self):
        """
        Returns a dictionary or vector of the entire game state.
        For RL, we will flatten this later.
        """
        return {
            "factories": self.factories.copy(),
            "center": self.center.copy(),
            "players": [p.get_state_vector() for p in self.players],
            "current_player": self.current_player_idx,
            "first_player_token": 1 if self.first_player_token_available else 0
        }