import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, device, chunk_size=5, discount=0.99):
        self.device = device
        self.chunk_size = chunk_size
        self.discount = discount
        self.size = 0

        # Online data storage
        self.online_states = []
        self.online_actions = []
        self.online_next_states = []
        self.online_rewards = []
        self.online_masks = []
        self.online_size = 0

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

    def add_chunk(self, state, action_chunk, next_state, rewards, mask):
        """Add a single online chunk transition.

        Args:
            state: obs at start of chunk (action_dim,)
            action_chunk: h actions flattened (action_dim * h,)
            next_state: obs after all h actions (obs_dim,)
            rewards: list of h rewards
            mask: 1.0 if episode alive at end, 0.0 if done
        """
        cum_reward = 0.0
        for i, r in enumerate(rewards):
            cum_reward += (self.discount ** i) * r

        self.online_states.append(state)
        self.online_actions.append(action_chunk)
        self.online_next_states.append(next_state)
        self.online_rewards.append([cum_reward])
        self.online_masks.append([mask])
        self.online_size += 1

    def _sample_offline(self, n):
        idx = self.valid_indices[np.random.randint(0, len(self.valid_indices), size=n)]
        h = self.chunk_size

        states = self.state[idx]
        action_chunks = np.concatenate(
            [self.action[idx + i] for i in range(h)], axis=-1
        )
        states_h = self.next_state[idx + h - 1]
        cum_reward = np.zeros((n, 1))
        for i in range(h):
            cum_reward += (self.discount ** i) * self.reward[idx + i]
        masks = self.mask[idx + h - 1]

        return states, action_chunks, states_h, cum_reward, masks

    def _sample_online(self, n):
        idx = np.random.randint(0, self.online_size, size=n)
        states = np.array([self.online_states[i] for i in idx])
        actions = np.array([self.online_actions[i] for i in idx])
        next_states = np.array([self.online_next_states[i] for i in idx])
        rewards = np.array([self.online_rewards[i] for i in idx])
        masks = np.array([self.online_masks[i] for i in idx])
        return states, actions, next_states, rewards, masks

    def sample(self, batch_size):
        if self.online_size == 0:
            # Pure offline
            s, a, ns, r, m = self._sample_offline(batch_size)
        else:
            # Half offline, half online
            n_online = batch_size // 2
            n_offline = batch_size - n_online
            s1, a1, ns1, r1, m1 = self._sample_offline(n_offline)
            s2, a2, ns2, r2, m2 = self._sample_online(n_online)
            s = np.concatenate([s1, s2])
            a = np.concatenate([a1, a2])
            ns = np.concatenate([ns1, ns2])
            r = np.concatenate([r1, r2])
            m = np.concatenate([m1, m2])

        return (
            torch.FloatTensor(s).to(self.device),
            torch.FloatTensor(a).to(self.device),
            torch.FloatTensor(ns).to(self.device),
            torch.FloatTensor(r).to(self.device),
            torch.FloatTensor(m).to(self.device),
        )
