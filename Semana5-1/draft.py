import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


rows, cols = 11, 11

rewards = np.full((rows, cols), -100)

valid_paths = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (2, 1),
    (2, 7),
    (2, 9),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 9),
    (4, 3),
    (4, 7),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (6, 5),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 5),
    (7, 6),
    (7, 7),
    (7, 8),
    (7, 9),
    (8, 3),
    (8, 7),
    (9, 0),
    (9, 1),
    (9, 2),
    (9, 3),
    (9, 4),
    (9, 5),
    (9, 6),
    (9, 7),
    (9, 8),
    (9, 9),
    (9, 10),
]
for r, c in valid_paths:
    rewards[r, c] = -1  # Valid paths with a reward of -1

rewards[0, 5] = 100  # Green goal cell


actions = ['up', 'down', 'left', 'right']
action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


q_table = np.zeros((rows, cols, len(actions)))


alpha = 0.01  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.9  # Exploration rate
episodes = 10000
max_steps = 100

def take_action(state, action):
    row, col = state
    dr, dc = action_map[action]
    new_row, new_col = row + dr, col + dc
    
    # Check boundaries
    if 0 <= new_row < rows and 0 <= new_col < cols:
        if rewards[new_row, new_col] == -100:  # Wall penalty
            return state, -100
        return (new_row, new_col), rewards[new_row, new_col]
    return state, -100  # Out of bounds

# Q-Learning algorithm
for episode in range(episodes):
    state = (10, 5)  # Starting state (bottom center)
    for step in range(max_steps):
        # Choose action (epsilon-greedy policy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]

        # Take action and observe reward and next state
        next_state, reward = take_action(state, action)
        
        # Update Q-value
        current_q = q_table[state[0], state[1], actions.index(action)]
        max_next_q = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], actions.index(action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)
        
        # Transition to next state
        state = next_state
        
        # If goal is reached, break
        if reward == 100:
            break

# Display the learned Q-values for each state
optimal_policy = np.chararray((rows, cols), unicode=True)
optimal_policy[:] = ' '

for i in range(rows):
    for j in range(cols):
        if rewards[i, j] == -100:
            optimal_policy[i, j] = 'X'  # Black cells
        elif rewards[i, j] == 100:
            optimal_policy[i, j] = 'G'  # Goal
        else:
            best_action = np.argmax(q_table[i, j])
            optimal_policy[i, j] = actions[best_action][0].upper()  # First letter of the action

# Display optimal policy as a DataFrame for better readability
optimal_policy_df = pd.DataFrame(optimal_policy)

# Display the policy
print("Optimal Policy for the Grid:")
print(optimal_policy_df)

# Visual representation using matplotlib
plt.figure(figsize=(8, 8))
for i in range(rows):
    for j in range(cols):
        if rewards[i, j] == -100:
            plt.text(
                j,
                i,
                "X",
                ha="center",
                va="center",
                color="white",
                bbox=dict(boxstyle="square", facecolor="black"),
            )
        elif rewards[i, j] == 100:
            plt.text(
                j,
                i,
                "G",
                ha="center",
                va="center",
                color="green",
                bbox=dict(boxstyle="square", facecolor="lightgreen"),
            )
        else:
            plt.text(
                j,
                i,
                optimal_policy[i, j],
                ha="center",
                va="center",
                bbox=dict(boxstyle="square", facecolor="white"),
            )

plt.xticks(range(cols))
plt.yticks(range(rows))
plt.grid(True)
plt.gca().invert_yaxis()
plt.title("Optimal Policy Visualization\n")
plt.tight_layout()
plt.show()
