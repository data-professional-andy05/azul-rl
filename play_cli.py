from src.agent.rl_env import AzulEnv
from src.azul.constants import PLAYABLE_COLORS, ID_TO_COLOR

def play_manual():
    # render_mode='human' enables the print statements
    env = AzulEnv(num_players=2, render_mode="human")
    obs, _ = env.reset()
    
    print("Welcome to Azul CLI!")
    print("You are controlling ALL players (Hotseat mode).")
    
    running = True
    while running:
        env.render()
        
        # Input Loop
        try:
            print("\nMake a Move:")
            source = int(input("  Source (0-4 Factories, 5 Center): "))
            
            print("  Colors: 0=Blue, 1=Yellow, 2=Red, 3=Black, 4=White")
            color_idx = int(input("  Color Index (0-4): "))
            
            dest = int(input("  Dest Row (0-4, 5 for Floor): "))
            
            # Encode Action
            # Formula: (Source * 30) + (Color_Index * 6) + Destination
            action = (source * 30) + (color_idx * 6) + dest
            
            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not info["valid"]:
                print("\n>>> INVALID MOVE! Try again. <<<")
            else:
                print(f"\n>>> Move Accepted. Reward: {reward}")
                
            if terminated:
                print("\nGAME OVER!")
                env.render()
                running = False
                
        except ValueError:
            print("Invalid input format. Please enter numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    play_manual()