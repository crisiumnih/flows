import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, device, chunk_size=5, discount=0.99):
        self.device = device
        self.chunk_size = chunk_size
        self.discount = discount
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

        # Find episode boundaries: terminals[i] = 1 means episode ends at i.
        terminals = dataset['terminals'].reshape(-1)
        terminal_indices = np.where(terminals > 0)[0]

        # For each index, compute which episode it belongs to and where that episode ends.
        # An index t is valid for chunk sampling if t+h-1 < episode_end (no crossing boundaries).
        episode_ends = np.zeros(self.size, dtype=np.int64)
        prev = 0
        for term_idx in terminal_indices:
            episode_ends[prev:term_idx + 1] = term_idx
            prev = term_idx + 1
        # Handle trailing data with no terminal
        if prev < self.size:
            episode_ends[prev:] = self.size - 1

        # Valid indices: those where we can grab h consecutive steps without crossing an episode boundary.
        # We need index t such that t + chunk_size - 1 <= episode_end[t]
        # (i.e., all h actions and the state at t+h are within the same episode)
        h = self.chunk_size
        self.valid_indices = np.where(
            (episode_ends - np.arange(self.size)) >= h
        )[0]

        print(f"Replay buffer: {self.size} transitions, {len(self.valid_indices)} valid chunk starts (h={h})")

    def sample(self, batch_size):
        # Sample from valid indices only
        idx = self.valid_indices[np.random.randint(0, len(self.valid_indices), size=batch_size)]

        h = self.chunk_size

        # Starting state
        states = self.state[idx]

        # Action chunks: h consecutive actions flattened
        # shape: (batch, action_dim * h)
        action_chunks = np.concatenate(
            [self.action[idx + i] for i in range(h)], axis=-1
        )

        # State after executing all h actions
        # next_state of the last action in the chunk = state at t+h
        states_h = self.next_state[idx + h - 1]

        # Discounted cumulative reward over the chunk
        # R = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{h-1}*r_{t+h-1}
        cum_reward = np.zeros((batch_size, 1))
        for i in range(h):
            cum_reward += (self.discount ** i) * self.reward[idx + i]

        # Mask at the end of the chunk (is the episode still alive?)
        masks = self.mask[idx + h - 1]

        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(action_chunks).to(self.device),
            torch.FloatTensor(states_h).to(self.device),
            torch.FloatTensor(cum_reward).to(self.device),
            torch.FloatTensor(masks).to(self.device),
        )
