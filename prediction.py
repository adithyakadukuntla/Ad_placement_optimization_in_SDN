from models.dqn_model import DQN
import torch
from environment.ad_env import AdPlacementEnv  # Import your environment

# Load the environment and model
env = AdPlacementEnv()  # Initialize the environment
state_size = env.state_space.shape[0]
action_size = env.action_space.n
model = DQN(state_size, action_size)

# Load the trained weights
model.load_state_dict(torch.load("models/dqn_model.pth"))
model.eval()  # Set model to evaluation mode

# Perform inference
state = env.reset()  # Get initial state from the environment
state_tensor = torch.FloatTensor(state)
with torch.no_grad():
    action = torch.argmax(model(state_tensor)).item()

# Display the recommended action
print(f"Recommended ad placement: {action}")


# D:\PProjects\ad-placement-rl\models\dqn_model.pth