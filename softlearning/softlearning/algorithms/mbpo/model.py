import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import gzip

device = torch.device('cuda')

num_train = 60000 # 60k train examples
num_test = 10000 # 10k test examples
train_inputs_file_path = './MNIST_data/train-images-idx3-ubyte.gz'
train_labels_file_path = './MNIST_data/train-labels-idx1-ubyte.gz'
test_inputs_file_path = './MNIST_data/t10k-images-idx3-ubyte.gz'
test_labels_file_path = './MNIST_data/t10k-labels-idx1-ubyte.gz'

BATCH_SIZE = 100

class Game_model(nn.Module):
    def __init__(self, network_size, state_size, action_size, reward_size, hidden_size=200, learning_rate=1e-2):
        super(Game_model, self).__init__()
        self.network_size = network_size
        self.hidden_size = hidden_size
        self.nn1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size * network_size),
            Swish()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_size * network_size, hidden_size * network_size),
            Swish()
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_size * network_size, hidden_size * network_size),
            Swish()
        )
        self.nn4 = nn.Sequential(
            nn.Linear(hidden_size * network_size, hidden_size * network_size),
            Swish()
        )

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = nn.Linear(hidden_size * network_size, self.output_dim * 2 * network_size)

        self.max_logvar = Variable(torch.ones((1, self.output_dim)).type(torch.FloatTensor) / 2, requires_grad=True).to(device)
        self.min_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 10, requires_grad=True).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        nn1_output = self.nn1(x)
        nn2_output = self.nn2(nn1_output)
        nn3_output = self.nn3(nn2_output)
        nn4_output = self.nn4(nn3_output)
        nn5_output = self.nn5(nn4_output)

        nn5_output = nn5_output.view(-1, self.network_size, self.output_dim * 2).transpose(0,1)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, torch.exp(logvar)

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim = 2), dim = 1).squeeze()
            var_loss = torch.mean(torch.mean(logvar, dim = 2), dim = 1).squeeze()
            total_loss = mse_loss + var_loss
        else:
            mse_loss = nn.MSELoss()
            total_loss = mse_loss(input=logits, target=labels)
        return total_loss, torch.mean(total_loss)

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()

class Ensemble_Model():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200):
        self.network_size = network_size
        self.elite_size = elite_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        self.model = Game_model(network_size, state_size, action_size, reward_size, hidden_size)

    def train(self, inputs, labels):
        inputs = torch.from_numpy(inputs).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

        mean, logvar = self.model(inputs)
        losses, total_loss = self.model.loss(mean, logvar, labels)
        losses = losses.detach().cpu().numpy()
        self.model.train(total_loss)

        sorted_loss_idx = np.argsort(losses)
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

    def predict(self, inputs):
        #TODO: change hardcode number to len(?)
        ensemble_mean = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        ensemble_logvar = np.zeros((self.network_size, inputs.shape[0], self.state_size + self.reward_size))
        inputs = torch.from_numpy(inputs).float().to(device)
        ensemble_mean, ensemble_logvar = self.model(inputs)
        ensemble_mean, ensemble_logvar = ensemble_mean.detach().cpu().numpy(), ensemble_logvar.detach().cpu().numpy()

        return ensemble_mean, ensemble_logvar


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

def get_data(inputs_file_path, labels_file_path, num_examples):
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)

def main():
    # Import MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels = get_data(train_inputs_file_path, train_labels_file_path, num_train)
    test_inputs, test_labels = get_data(test_inputs_file_path, test_labels_file_path, num_test)

    model = Ensemble_Model(5, 3, 5, 779, 5, 50)
    for i in range(0, 10000, BATCH_SIZE):
        model.train(Variable(torch.from_numpy(train_inputs[i:i+BATCH_SIZE])), Variable(torch.from_numpy(train_labels[i:i+BATCH_SIZE])))
    model.predict(Variable(torch.from_numpy(test_inputs[:1000])))

if __name__ == '__main__':
    main()
