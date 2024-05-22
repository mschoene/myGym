from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines_mygym.ppo2 import PPO2
import torch
import torch.nn as nn

class MassDistributionNN(nn.Module):
    def __init__(self, observation_space, action_space, output_dim, fc_output_size=128, lstm_hidden_size=256, **kwargs):
        super(MassDistributionNN, self).__init__(observation_space, action_space, **kwargs)
        self.fc1 = nn.Linear(observation_space + action_space.shape[-1], fc_output_size)
        self.fcq_0 = nn.Linear(observation_space, fc_output_size)
        self.lstm = nn.LSTM(input_size=fc_output_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.features_dim, output_dim)

    def forward(self, observation, initial_observation, past_action):
        obs = torch.cat((observation, past_action), dim=-1)
        x = torch.relu(self.fc1(obs))
        y = self.fcq_0(initial_observation)

        if self.lstm_hidden_state is None:
            self.lstm_hidden_state = (torch.zeros(1, x.size(0), self.lstm_hidden_size),
                                      torch.zeros(1, x.size(0), self.lstm_hidden_size))
        lstm_input = torch.cat((x, y), dim=0)

        x, self.lstm_hidden_state = self.lstm(lstm_input, self.lstm_hidden_state)
        x = torch.relu(self.fc2(x[-1, :, :]))
        return x
    

class MassPPOPolicy(PPO2):
    def __init__(self,  policy, env, observation_space, action_space, mass_output_dim=3, **kwargs):
        super(MassPPOPolicy, self).__init__(policy, env)
        self.mass_distribution_nn = MassDistributionNN(input_dim=observation_space.shape[-1], action_space=action_space, output_dim=mass_output_dim)
        self.fc = nn.Linear(observation_space + mass_output_dim, observation_space)

    def set_initial_observation(self, init_obs):
        self.init_obs = init_obs
        
    def forward(self, obs, past_action, deterministic=False):

        mass_distribution = self.mass_distribution_nn(obs, self.init_obs, past_action)
        new_obs = torch.cat((obs, mass_distribution), dim=-1)
        new_obs = self.fc(new_obs)
        
        return super(MassPPOPolicy, self).forward(new_obs, deterministic)