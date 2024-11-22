from environment.ad_env import AdPlacementEnv
from models.dqn_model import train_dqn, plot_rewards , calculate_action_rewards
import torch

if __name__ == "__main__":
    env = AdPlacementEnv()
    print("Training RL agent for Ad Placement Optimization...")
    
    # Train the model and get total_rewards
    model, total_rewards, actions_taken = train_dqn(env)
    print("Training complete!")

    # Plot rewards
    plot_rewards(total_rewards)
    calculate_action_rewards(actions_taken, total_rewards, env.action_space.n)
    
    # Save the trained model
    torch.save(model.state_dict(), "dqn_model.pth")
    print("Model saved to dqn_model.pth")
    
    # Save total_rewards to a txt file
    with open("total_rewards.txt", "w") as file:
        for reward in total_rewards:
            file.write(f"{reward}\n")  # Write each reward on a new line

    print("Total rewards saved to total_rewards.txt")
