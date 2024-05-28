import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import TensorDataset
from stable_baselines_mygym.massPpo import MassDistributionNN
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import json
import csv
import ast
from torch.nn.utils.rnn import pad_sequence

class CubeDataset(Dataset):
    
    def __init__(self, episodes, coms, sequence_length, device):
        self.episodes = episodes
        self.longest_episode = max([len(x) for x in episodes])
        self.coms = coms
        self.sequence_length = sequence_length
        self.device = device

    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        com = self.coms[idx]

        start_idx = 0
        end_idx = self.sequence_length

        while len(episode) < self.longest_episode:
            episode.append([0,0,0,0,0,0,0])
        if len(episode) >= self.sequence_length:
            sequences = []
            while end_idx <= len(episode):
                sequences.append(episode[start_idx:end_idx])
                start_idx += 1
                end_idx = start_idx + self.sequence_length
                
            sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)

        return sequences, torch.tensor(com, dtype=torch.float32).to(self.device)


if __name__ == "__main__":
    config_path = './stable_baselines_mygym/lstm_config.json'
    data_path = './dataset_com_lstm.csv'
    with open(config_path, 'r') as config_file:
        c = json.load(config_file)
    with open(data_path, 'r') as data_file:
        data = pd.read_csv(data_file)
    
    data.iloc[:,0] = data.iloc[:,0].apply(ast.literal_eval)
    data.iloc[:,1] = data.iloc[:,1].apply(ast.literal_eval)


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    y = data.iloc[:, 1]
    x = data.iloc[:, 0]


    dataset = CubeDataset(x, y, c["sequence_length"], device)


    train_size = int(0.9 * len(dataset))  # 90% for training
    eval_size = len(dataset) - train_size  # Remaining 10% for evaluation

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, batch_size=c["batch_size"], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=c["batch_size"], shuffle=False)

    model = MassDistributionNN(c["input_dim"], c["output_dim"], c["hidden_dim"], c["sequence_length"]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses =[]
    eval_losses = []
    for epoch in range(c["num_epochs"]):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_dataloader:

            batch_x, batch_y = batch_x.to(torch.device(device)), batch_y.to(torch.device(device))

            outputs = model(batch_x)

            batch_y = batch_y.unsqueeze(1).repeat(1, batch_x.size(1), 1).view(-1, c["output_dim"])

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

        print(f'Epoch [{epoch+1}/{c["num_epochs"]}], Loss: {epoch_train_loss/len(train_dataloader.dataset):.7f}')

        #EVAL
        model.eval()
        epoch_eval_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_dataloader:
                inputs, targets = inputs.to(torch.device(device)), targets.to(torch.device(device))
                targets = targets.unsqueeze(1).repeat(1, inputs.size(1), 1).view(-1, c["output_dim"])

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_eval_loss += loss.item() * inputs.size(0)
        epoch_eval_loss /= len(eval_dataloader.dataset)
        eval_losses.append(epoch_eval_loss)

        print(f'Epoch [{epoch+1}/{c["num_epochs"]}], '
            f'Training Loss: {epoch_train_loss:.4f}, '
            f'Evaluation Loss: {epoch_eval_loss:.4f}')
    with open("./lstm_loss_results.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eval_losses)
        writer.writerow(train_losses)
    print("Training complete")
    torch.save(model.state_dict(), "lstm_model.pth")
