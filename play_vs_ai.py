import sys
import os
import time
import numpy as np
from sb3_contrib import MaskablePPO
from src.agent.rl_env import AzulEnv
from src.azul.constants import ID_TO_COLOR

# CHANGE THIS TO YOUR MODEL PATH
MODEL_PATH = "models/ppo_azul_big_1M/azul_1M_final.zip"
LOG_FILE = "game_debug_log.txt"

def log(msg, to_file=True):
    """Prints to console AND writes to file."""
    print(msg)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def print_board(env):
    game = env.game
    log("\n" + "="*60)
    log(f" ROUND {game.round_number} | TURN: Player {game.current_player_idx}")
    log("="*60)
    
    log("\n🏭 FACTORIES:")
    for i, f in enumerate(game.factories):
        tiles = []
        for color_id, count in enumerate(f):
            if color_id == 0: continue 
            if count > 0: tiles.extend([ID_TO_COLOR[color_id]] * count)
        log(f"  [{i}] {', '.join(tiles) if tiles else 'EMPTY'}")
        
    log("\n♻️  CENTER:")
    center_tiles = []
    for color_id, count in enumerate(game.center):
        if color_id == 0: continue
        if count > 0: center_tiles.extend([ID_TO_COLOR[color_id]] * count)
    log(f"  {', '.join(center_tiles) if center_tiles else 'EMPTY'}  {'[Start Token]' if game.first_player_token_available else ''}")
    
    log("\n" + "-"*60)
    for p_idx, p in enumerate(game.players):
        marker = "👉" if p_idx == game.current_player_idx else "  "
        name = "YOU (P0)" if p_idx == 0 else "AI (P1) "
        log(f"\n{marker} {name} | Score: {p.score}")
        log("    Patterns         | Wall")
        log("    -----------------|---------------------")
        for r in range(5):
            pat_col = p.pattern_lines_color[r]
            pat_cnt = p.pattern_lines_count[r]
            pat_sym = ID_TO_COLOR[pat_col] if pat_col != 0 else "."
            pat_str = (pat_sym * pat_cnt).rjust(5)
            wall_row = [ID_TO_COLOR[p.wall[r, c]] if p.wall[r, c] != 0 else "." for c in range(5)]
            log(f"    {pat_str} /{r+1}      | {' '.join(wall_row)}")
        floor_tiles = [ID_TO_COLOR[t] for t in p.floor_line if t != 0]
        log(f"    Floor Line: [{', '.join(floor_tiles)}] (Penalty: {p.floor_line_count})")

def play():
    # Reset Log File
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("--- AZUL GAME LOG ---\n")

    log(f"Loading Brain from: {MODEL_PATH}...")
    try:
        model = MaskablePPO.load(MODEL_PATH)
    except FileNotFoundError:
        log("❌ Model file not found!")
        return

    env = AzulEnv(num_players=2) 
    obs, _ = env.reset()
    
    # Enable verbose logging in the game engine
    env.game.round_logs = {}
    
    running = True
    while running:
        # WE REMOVED clear_screen() SO YOU CAN SCROLL UP
        print_board(env)
        
        prev_round = env.game.round_number
        curr_player = env.game.current_player_idx
        
        # --- PLAYER INPUT / AI LOGIC ---
        if curr_player == 0:
            log("\n>>> 🧠 YOUR TURN (Player 0) <<<")
            valid = False
            while not valid:
                try:
                    user_input = input("\nEnter Move (Source Color Dest): ").strip().split()
                    
                    # Log what the user typed so we can debug input errors too
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"USER INPUT: {user_input}\n")
                        
                    if len(user_input) != 3: continue
                    s, c, d = map(int, user_input)
                    action = (s * 30) + (c * 6) + d
                    mask = env.action_masks()
                    if not mask[action]: 
                        log("❌ ILLEGAL MOVE!")
                        continue
                    obs, reward, terminated, _, _ = env.step(action)
                    valid = True
                except ValueError: pass
        else:
            log("\n>>> 🤖 AI TURN (Player 1) <<<")
            # Deterministic=True makes the AI play its best move
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            s, c, d = env.decode_action(action)
            c_str = list(ID_TO_COLOR.values())[c+1] 
            log(f"🤖 AI plays: S{s} C{c_str} -> D{d}")

        # --- SCORING REPORT (Detect Round Change) ---
        curr_round = env.game.round_number
        
        # Logic: If round number increased OR game ended
        if curr_round > prev_round or terminated:
            log("\n" + "*"*50)
            log(f"📊 SCORING REPORT FOR ROUND {prev_round}")
            log("*"*50)
            
            for p_idx in [0, 1]:
                name = "YOU" if p_idx == 0 else "AI "
                log(f"\nPlayer {name} Score Actions:")
                if p_idx in env.game.round_logs:
                    for log_msg in env.game.round_logs[p_idx]:
                        log(f"  > {log_msg}")
                else:
                    log("  (No tiles scored)")
                log(f"  New Total Score: {env.game.players[p_idx].score}")

            input("\nPress Enter to continue (Check logs now if you want)...")

        # --- GAME OVER ---
        if terminated:
            log("\nGAME OVER")
            p0 = env.game.players[0].score
            p1 = env.game.players[1].score
            log(f"Final Score: YOU {p0} - AI {p1}")
            
            if p0 > p1: log("🏆 YOU WIN!")
            elif p1 > p0: log("💀 AI WINS!")
            else: log("Draw")
            
            running = False

if __name__ == "__main__":
    play()