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
        """Resets the board for a new game."""
        self.score = 0
        self.wall = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        # Pattern Lines
        self.pattern_lines_color = np.zeros(GRID_SIZE, dtype=np.int8)
        self.pattern_lines_count = np.zeros(GRID_SIZE, dtype=np.int8)
        
        # Floor line
        self.floor_line = np.zeros(FLOOR_LINE_CAPACITY, dtype=np.int8)
        self.floor_line_count = 0

    def get_row_capacity(self, row_idx):
        return row_idx + 1

    def can_add_to_pattern_line(self, row_idx, color):
        # Rule 3: Check Wall
        target_col = np.where(WALL_PATTERN[row_idx] == color)[0][0]
        if self.wall[row_idx, target_col] != EMPTY:
            return False

        # Rule 1 & 2: Check Pattern Line
        current_color = self.pattern_lines_color[row_idx]
        current_count = self.pattern_lines_count[row_idx]
        capacity = self.get_row_capacity(row_idx)

        if current_count >= capacity:
            return False
        
        if current_color != EMPTY and current_color != color:
            return False
            
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

    # --- SCORING LOGIC STARTS HERE ---

    def calculate_round_bonuses(self):
        """
        Executes end-of-round logic:
        1. Moves full pattern lines to Wall.
        2. Calculates score (Horizontal + Vertical).
        3. Applies floor penalties.
        4. Resets floor line and full pattern lines.
        """
        round_score = 0

        # 1. Process Pattern Lines
        for row in range(GRID_SIZE):
            capacity = self.get_row_capacity(row)
            
            # Only process if row is full
            if self.pattern_lines_count[row] == capacity:
                color = self.pattern_lines_color[row]
                
                # Find where this color goes on the wall
                col = np.where(WALL_PATTERN[row] == color)[0][0]
                
                # Move to Wall
                self.wall[row, col] = color
                
                # Calculate placement score
                round_score += self._calculate_placement_score(row, col)
                
                # Reset this pattern line (it's now empty)
                self.pattern_lines_count[row] = 0
                self.pattern_lines_color[row] = EMPTY
            
            # If not full, tiles stay for next round (standard Azul rule)

        # 2. Subtract Floor Line Penalty
        penalty = 0
        for i in range(self.floor_line_count):
            penalty += FLOOR_LINE_SCORES[i]
        
        round_score += penalty
        
        # Score cannot go below 0 in total game score? 
        # Actually in Azul, score *can* go down, but usually usually floored at 0 for the game total.
        # We'll allow negative step rewards for RL, but keep self.score >= 0 logic if strict rules desired.
        self.score += round_score
        if self.score < 0: self.score = 0

        # 3. Clear Floor Line
        self.floor_line.fill(EMPTY)
        self.floor_line_count = 0
        
        return round_score

    def _calculate_placement_score(self, row, col):
        """
        Calculates points for a single tile placed at (row, col).
        """
        # Horizontal Score
        # We start with 0. If there are neighbors, we count the whole line.
        # If no neighbors, this block stays 0.
        horiz_points = 0
        
        # Check Left
        c = col - 1
        while c >= 0 and self.wall[row, c] != EMPTY:
            horiz_points += 1
            c -= 1
            
        # Check Right
        c = col + 1
        while c < GRID_SIZE and self.wall[row, c] != EMPTY:
            horiz_points += 1
            c += 1
            
        if horiz_points > 0:
            horiz_points += 1 # Add the tile itself

        # Vertical Score
        vert_points = 0
        
        # Check Up
        r = row - 1
        while r >= 0 and self.wall[r, col] != EMPTY:
            vert_points += 1
            r -= 1
            
        # Check Down
        r = row + 1
        while r < GRID_SIZE and self.wall[r, col] != EMPTY:
            vert_points += 1
            r += 1
            
        if vert_points > 0:
            vert_points += 1 # Add the tile itself

        # Final Calculation
        if horiz_points == 0 and vert_points == 0:
            return 1 # Just the tile itself
        
        # If we have both, we sum them (the tile counts twice, which is correct rules)
        # If we only have horizontal, vert is 0.
        # Note: If horiz > 0, the tile is included in horiz_points.
        return max(horiz_points, 0) + max(vert_points, 0)
        
    def get_state_vector(self):
        """Flattens board state for the AI."""
        return np.concatenate([
            self.wall.flatten(),
            self.pattern_lines_color,
            self.pattern_lines_count,
            self.floor_line
        ])