import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')

BATCH_SIZE = 1000

class Game_model(nn.Module):
    def __init__(self, state_size, action_size, reward_size, hidden_size=200, learning_rate=1e-2):
        super(Game_model, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            Swish()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )
        self.nn4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()
        )
        self.nn5 = nn.Linear(hidden_size, state_size + reward_size)

        self.optimizer = torch.optim.Adam(lr=learning_rate)

    def forward(self, x):
        nn1_output = self.nn1(x)
        nn2_output = self.nn2(nn1_output)
        nn3_output = self.nn3(nn2_output)
        nn4_output = self.nn4(nn3_output)
        nn5_output = self.nn5(nn4_output)


    def loss(self, logits, labels):
        mse_loss = nn.MSELoss()
        loss = mse_loss(input=logits, target=labels)
        return loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Ensemble_Model():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size, hidden_size):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        for i in range(network_size):
            model_list.append(Game_model(state_size, action_size, reward_size, hidden_size))

    def train(self, inputs, labels):
        losses = []
        for model in model_list:
            logits = model(inputs)
            loss = model.loss(logits, labels)
            model.train(loss)
            losses.append(loss)
        sorted_loss_idx = np.argsort(losses)
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

    def predict(self, inputs):
        ensemble_logtis = np.zeros((self.state_size + self.reward_size, self.elite_size))
        cnt = 0
        for idx in elite_model_idxes:
            ensemble_logtis[:cnt] = model[idx](inputs).detach().numpy()
            cnt += 1
        return ensemble_logtis


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
