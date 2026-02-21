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
        """
        Executes end-of-round logic.
        Returns: 
            points (int): The score for this round.
            discarded_tiles (list): List of color IDs to return to the box.
            logs (list): Text description of scoring events.
        """
        round_score = 0
        discarded_tiles = []
        logs = []

        # 1. Process Pattern Lines (Top to Bottom)
        for row in range(GRID_SIZE):
            capacity = self.get_row_capacity(row)
            
            if self.pattern_lines_count[row] == capacity:
                color = self.pattern_lines_color[row]
                col = np.where(WALL_PATTERN[row] == color)[0][0]
                
                # A. Move to Wall
                self.wall[row, col] = color
                
                # B. Calculate Score
                pts, log_msg = self._calculate_placement_score(row, col, return_log=True)
                round_score += pts
                if verbose: 
                    logs.append(f"Row {row} (Col {col}): {log_msg}")
                
                # C. Recycle Remaining Tiles
                num_discarded = capacity - 1
                if num_discarded > 0:
                    discarded_tiles.extend([color] * num_discarded)
                
                # D. Reset Pattern Line
                self.pattern_lines_count[row] = 0
                self.pattern_lines_color[row] = EMPTY
            
        # 2. Process Floor Line Logic
        penalty = 0
        limit = min(self.floor_line_count, len(FLOOR_LINE_SCORES)) # Use len to avoid index error
        
        for i in range(limit):
             penalty += FLOOR_LINE_SCORES[i]
        
        # If overflowing beyond defined scores, use max penalty (-3)
        if self.floor_line_count > len(FLOOR_LINE_SCORES):
             penalty += (self.floor_line_count - len(FLOOR_LINE_SCORES)) * FLOOR_LINE_SCORES[-1]

        if penalty != 0 and verbose:
            logs.append(f"Floor Line Penalty: {penalty} ({self.floor_line_count} tiles)")

        round_score += penalty
        
        # Recycle Floor Tiles
        limit_recycle = min(self.floor_line_count, FLOOR_LINE_CAPACITY)
        for i in range(limit_recycle):
            tile = self.floor_line[i]
            if tile != EMPTY and tile != 6: 
                discarded_tiles.append(tile)

        # Apply Score (Floor at 0)
        self.score += round_score
        if self.score < 0: self.score = 0

        # Clear Floor Line
        self.floor_line.fill(EMPTY)
        self.floor_line_count = 0
        
        return round_score, discarded_tiles, logs

    def calculate_end_game_score(self):
        bonus = 0
        # Rows
        for r in range(GRID_SIZE):
            if np.count_nonzero(self.wall[r]) == GRID_SIZE: bonus += 2
        # Columns
        for c in range(GRID_SIZE):
            if np.count_nonzero(self.wall[:, c]) == GRID_SIZE: bonus += 7
        # Colors
        for color in range(1, 6): 
            if np.count_nonzero(self.wall == color) == GRID_SIZE: bonus += 10
                
        self.score += bonus
        return bonus

    def _calculate_placement_score(self, row, col, return_log=False):
        # Horizontal Check
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
            
        # If neighbors found, add self (total length)
        # If no neighbors, horiz_points is 0
        if horiz_points > 0: horiz_points += 1 

        # Vertical Check
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
            
        if vert_points > 0: vert_points += 1 

        # Final Calculation
        total = 0
        if horiz_points == 0 and vert_points == 0:
            total = 1 
        else:
            # If both directions have points, the tile counts twice (once in each sum)
            # Example:  X
            #         Y Z A
            #           B
            # Placing Z (at center):
            # Horiz: Y-Z-A (3)
            # Vert: X-Z-B (3)
            # Total: 6
            total = max(horiz_points, 0) + max(vert_points, 0)
            
        if return_log:
            return total, f"Placed. Horiz_Chain: {horiz_points}, Vert_Chain: {vert_points}. Points: {total}"
        return total

    def get_state_vector(self):
        return np.concatenate([
            self.wall.flatten(),
            self.pattern_lines_color,
            self.pattern_lines_count,
            self.floor_line
        ])