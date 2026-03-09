import numpy as np
from .constants import (
    GRID_SIZE, 
    EMPTY, 
    FLOOR_LINE_CAPACITY, 
    FLOOR_LINE_SCORES, 
    WALL_PATTERN
)

class PlayerBoard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.wall = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.pattern_lines_color = np.zeros(GRID_SIZE, dtype=np.int8)
        self.pattern_lines_count = np.zeros(GRID_SIZE, dtype=np.int8)
        self.floor_line = np.zeros(FLOOR_LINE_CAPACITY, dtype=np.int8)
        self.floor_line_count = 0

    def get_row_capacity(self, row_idx):
        return row_idx + 1

    def can_add_to_pattern_line(self, row_idx, color):
        target_col = np.where(WALL_PATTERN[row_idx] == color)[0][0]
        if self.wall[row_idx, target_col] != EMPTY: return False

        current_color = self.pattern_lines_color[row_idx]
        current_count = self.pattern_lines_count[row_idx]
        capacity = self.get_row_capacity(row_idx)

        if current_count >= capacity: return False
        if current_color != EMPTY and current_color != color: return False
        return True

    def add_tiles(self, row_idx, color, count):
        if row_idx == -1:
            self._add_to_floor_line(color, count)
            return True

        if not self.can_add_to_pattern_line(row_idx, color):
            return False

        capacity = self.get_row_capacity(row_idx)
        current_count = self.pattern_lines_count[row_idx]
        
        if self.pattern_lines_color[row_idx] == EMPTY:
            self.pattern_lines_color[row_idx] = color
            
        space_remaining = capacity - current_count
        
        if count <= space_remaining:
            self.pattern_lines_count[row_idx] += count
        else:
            placed = space_remaining
            overflow = count - space_remaining
            self.pattern_lines_count[row_idx] += placed
            self._add_to_floor_line(color, overflow)
            
        return True

    def _add_to_floor_line(self, color, count):
        for _ in range(count):
            if self.floor_line_count < FLOOR_LINE_CAPACITY:
                self.floor_line[self.floor_line_count] = color
                self.floor_line_count += 1

    def calculate_round_bonuses(self, verbose=False):
        round_score = 0
        discarded_tiles = []
        logs = []

        for row in range(GRID_SIZE):
            capacity = self.get_row_capacity(row)
            if self.pattern_lines_count[row] == capacity:
                color = self.pattern_lines_color[row]
                col = np.where(WALL_PATTERN[row] == color)[0][0]
                
                self.wall[row, col] = color
                # Pass self.wall implicitly
                pts, log_msg = self._calculate_placement_score(row, col, self.wall, return_log=True)
                round_score += pts
                if verbose: logs.append(f"Row {row}: {log_msg}")
                
                num_discarded = capacity - 1
                if num_discarded > 0:
                    discarded_tiles.extend([color] * num_discarded)
                
                self.pattern_lines_count[row] = 0
                self.pattern_lines_color[row] = EMPTY
            
        penalty = 0
        limit = min(self.floor_line_count, len(FLOOR_LINE_SCORES))
        for i in range(limit): penalty += FLOOR_LINE_SCORES[i]
        
        if self.floor_line_count > len(FLOOR_LINE_SCORES):
             penalty += (self.floor_line_count - len(FLOOR_LINE_SCORES)) * FLOOR_LINE_SCORES[-1]

        if penalty != 0 and verbose: logs.append(f"Floor Penalty: {penalty}")

        round_score += penalty
        
        limit_recycle = min(self.floor_line_count, FLOOR_LINE_CAPACITY)
        for i in range(limit_recycle):
            tile = self.floor_line[i]
            if tile != EMPTY and tile != 6: discarded_tiles.append(tile)

        self.score += round_score
        if self.score < 0: self.score = 0

        self.floor_line.fill(EMPTY)
        self.floor_line_count = 0
        
        return round_score, discarded_tiles, logs

    def calculate_end_game_score(self):
        bonus = 0
        for r in range(GRID_SIZE):
            if np.count_nonzero(self.wall[r]) == GRID_SIZE: bonus += 2
        for c in range(GRID_SIZE):
            if np.count_nonzero(self.wall[:, c]) == GRID_SIZE: bonus += 7
        for color in range(1, 6): 
            if np.count_nonzero(self.wall == color) == GRID_SIZE: bonus += 10
        self.score += bonus
        return bonus

    # --- THE FULLY SIMULATED SCORE ---
    def get_complete_virtual_score(self):
        """
        Simulates the end of the round AND the end of the game.
        Returns: Current Score + Immediate Placement Points + Penalties + End Game Bonuses.
        """
        v_wall = self.wall.copy()
        v_score = self.score
        
        # 1. Simulate Placement Points (Waterfall)
        for row in range(GRID_SIZE):
            capacity = self.get_row_capacity(row)
            if self.pattern_lines_count[row] == capacity:
                color = self.pattern_lines_color[row]
                col = np.where(WALL_PATTERN[row] == color)[0][0]
                
                # Update Virtual Wall
                v_wall[row, col] = color
                
                # Calculate Points using Virtual Wall
                pts = self._calculate_placement_score(row, col, v_wall)
                v_score += pts

        # 2. Simulate Floor Penalty
        penalty = 0
        limit = min(self.floor_line_count, len(FLOOR_LINE_SCORES))
        for i in range(limit): penalty += FLOOR_LINE_SCORES[i]
        
        if self.floor_line_count > len(FLOOR_LINE_SCORES):
             penalty += (self.floor_line_count - len(FLOOR_LINE_SCORES)) * FLOOR_LINE_SCORES[-1]
             
        v_score += penalty
        if v_score < 0: v_score = 0
        
        # 3. Add End Game Bonuses (Based on Virtual Wall)
        bonus = 0
        for r in range(GRID_SIZE):
            if np.count_nonzero(v_wall[r]) == GRID_SIZE: bonus += 2
        for c in range(GRID_SIZE):
            if np.count_nonzero(v_wall[:, c]) == GRID_SIZE: bonus += 7
        for color in range(1, 6): 
            if np.count_nonzero(v_wall == color) == GRID_SIZE: bonus += 10
            
        return v_score + bonus

    def _calculate_placement_score(self, row, col, wall_state, return_log=False):
        """
        Calculates adjacency score.
        Args:
            wall_state: The matrix to check against (Real or Virtual).
        """
        # Horizontal
        horiz_points = 0
        c = col - 1
        while c >= 0 and wall_state[row, c] != EMPTY:
            horiz_points += 1
            c -= 1
        c = col + 1
        while c < GRID_SIZE and wall_state[row, c] != EMPTY:
            horiz_points += 1
            c += 1
        if horiz_points > 0: horiz_points += 1 

        # Vertical
        vert_points = 0
        r = row - 1
        while r >= 0 and wall_state[r, col] != EMPTY:
            vert_points += 1
            r -= 1
        r = row + 1
        while r < GRID_SIZE and wall_state[r, col] != EMPTY:
            vert_points += 1
            r += 1
        if vert_points > 0: vert_points += 1 

        total = 0
        if horiz_points == 0 and vert_points == 0: total = 1 
        else: total = max(horiz_points, 0) + max(vert_points, 0)
            
        if return_log: return total, f"Points: {total}"
        return total

    def get_state_vector(self):
        return np.concatenate([
            self.wall.flatten(),
            self.pattern_lines_color,
            self.pattern_lines_count,
            self.floor_line
        ])