import torch
import torch.nn as nn


class MassDistributionNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=3, hidden_dim=128, num_layer=5, **kwargs):
        super(MassDistributionNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layer, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # obs = torch.cat((observation, past_action), dim=-1)

        batch_size, num_sequences, seq_len, input_dim = x.size()
        x = x.view(batch_size*num_sequences * seq_len, input_dim)
        x = torch.relu(self.fc1(x))

        h0 = torch.zeros(self.lstm.num_layers, batch_size * num_sequences, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size * num_sequences, self.lstm.hidden_size).to(x.device)

        # x, self.lstm_hidden_state = self.lstm(x, self.lstm_hidden_state)
        out, _ = self.lstm(x.view(batch_size * num_sequences, seq_len, -1), (h0, c0))
        out = out[:, -1, :]
        out = self.fc2(out)
        return out
