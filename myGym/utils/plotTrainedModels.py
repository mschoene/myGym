import json
import os
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt

do_mlp = False

base_path = 'trained_models/pnp_franka_weighing_MLP_fromScratch/pnp_table_panda_step_ppo_'
xxx_values = ['2', '3', '4']  # Replace with your actual values

if not do_mlp:
    base_path = 'trained_models/pnp_franka_weighing_withCOMloss_relCOM_60fps_lastOne/pnp_table_panda_step_ppoRecurrCOM'
    xxx_values = ['', '_2', '_3']  # Replace with your actual values

# Initialize lists to store the accumulated data
episodes = []
mean_rewards = []
success_rates = []
std_rewards = []
# Initialize a variable to keep track of the cumulative episode number
cumulative_episode = 0
last_ep = 0

# Loop through the specified paths
for xxx in xxx_values:
    json_path = os.path.join(base_path + xxx, 'evaluation_results.json')
    if os.path.isfile(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                episode = int(value['episode'])
                cumulative_episode =  episode + last_ep
                mean_reward = float(value['mean_reward'])
                std_reward = float(value['std_reward'])
                success_rate = float(value['mean subgoals finished']) / 100. * 4.
                
                episodes.append(cumulative_episode)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                success_rates.append(success_rate)
            last_ep = cumulative_episode  # Update last_ep after processing all episodes


# Plotting the data
plt.figure(figsize=(12, 6))

# Plot mean reward vs. episodes
plt.subplot(1, 2, 1)
plt.plot(episodes, mean_rewards, marker='o', linestyle='-', color='b', label='Mean Reward')
plt.fill_between(episodes, [m - s for m, s in zip(mean_rewards, std_rewards)], 
                 [m + s for m, s in zip(mean_rewards, std_rewards)], color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Environment steps')
plt.ylabel('Mean Reward')
plt.title('Mean Reward vs training steps')
plt.grid(True)
plt.legend()

#plt.savefig('mean_reward_vs_episodes.png')

# Plot success rate vs. episodes
plt.subplot(1, 2, 2)
plt.plot(episodes, success_rates, marker='o', linestyle='-', color='r', label='Subtask completion averaged over 20 episodes')
plt.xlabel('Environment steps')
plt.ylabel('Average subtask completion')
plt.title('Subtask completion vs training steps')
plt.ylim(0, 4)
plt.grid(True)
plt.legend()

if do_mlp:
    plt.savefig('success_rate_vs_episodes.png')
else:
    plt.savefig('success_rate_vs_episodes_lstm_withCOMloss.png')

# Show the plots
#plt.tight_layout()
#plt.show()
