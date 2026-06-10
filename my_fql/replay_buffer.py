import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, device):
        self.device = device
        self.size = 0

    def load_ogbench(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.mask = dataset['masks'].reshape(-1, 1)

        if 'rewards' in dataset:
            self.reward = dataset['rewards'].reshape(-1, 1)
        else:
            self.reward = np.zeros((len(self.state), 1))

        self.size = self.state.shape[0]

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.mask[ind]).to(self.device),
        )
