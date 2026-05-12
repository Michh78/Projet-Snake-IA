import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.state_dict(), os.path.join(model_folder_path, file_name))

    def load(self, file_name='model.pth'):
        file_path = os.path.join('./model', file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=DEVICE))
            self.eval()
            print("Modèle chargé !")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long).to(DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(DEVICE)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        # Vectorised Bellman update — un seul forward pass pour tout le batch
        with torch.no_grad():
            next_q = self.model(next_state).max(dim=1)[0]

        done_tensor = torch.tensor(done, dtype=torch.bool).to(DEVICE)
        Q_new = reward + self.gamma * next_q * (~done_tensor)

        action_indices = torch.argmax(action, dim=1)
        target[torch.arange(len(done)), action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
