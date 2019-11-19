import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

BATCH_SIZE = 1000

class Game_model(nn.Module):
    def __init__(self, state_size, action_size, reward_size, hidden_size=200):
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


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

