import torch
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        nn1_output = self.nn1(x)
        nn2_output = self.nn2(nn1_output)
        nn3_output = self.nn3(nn2_output)
        nn4_output = self.nn4(nn3_output)
        nn5_output = self.nn5(nn4_output)
        return nn5_output

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
            self.model_list.append(Game_model(state_size, action_size, reward_size, hidden_size))

    def train(self, inputs, labels):
        losses = []
        for model in self.model_list:
            logits = model(inputs)
            loss = model.loss(logits, labels)
            model.train(loss)
            losses.append(loss)
        sorted_loss_idx = np.argsort(losses)
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

    def predict(self, inputs):
        #TODO: change hardcode number to len(?)
        ensemble_logtis = np.zeros((1000, self.state_size + self.reward_size, self.elite_size))
        cnt = 0
        for idx in self.elite_model_idxes:
            ensemble_logtis[:,:,cnt] = self.model_list[idx](inputs).detach().numpy()
            cnt += 1
        return ensemble_logtis


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
