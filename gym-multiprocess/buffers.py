import numpy as np
import random
import torch
import copy
import os
import pickle
from utils import get_device, get_space_shape



'''
    Replay buffer class (can be used for both on-policy and off-policy algorithms).
'''
class ReplayBuffer(object):
    def __init__(self, capacity, device = 'auto', seed = 123456) -> None:
        random.seed(seed)
        self.capacity = capacity
        self.pos = 0
        self.device = get_device(device)
        self.buffer = []
        self.sample_helper_func = lambda x: torch.from_numpy(np.concatenate(x)).float().to(self.device)


    def reset(self):
        self.buffer.clear()
        self.pos = 0
        

    def add(self, batch):
        # batch: arrays of size (n_env, ...)
        batch = copy.deepcopy(batch)
        self.pos = (self.pos + 1) % self.capacity
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.pos] = batch


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return tuple(map(self.sample_helper_func, zip(*batch)))


    def generate_dataset(self, batch_size = None):
        indices = np.random.permutation(self.size)
        if batch_size is None:
            batch_size = self.size
        start_idx = 0
        while start_idx < self.size:
            batch = [self.buffer[idx] for idx in indices[start_idx:start_idx + batch_size]]
            yield tuple(map(self.sample_helper_func, zip(*batch)))
            start_idx += batch_size


    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)


    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.buffer = pickle.load(f)
            self.pos = len(self.buffer) % self.capacity


    @property
    def size(self):
        return len(self.buffer)


    @property
    def is_full(self):
        return len(self.buffer) >= self.capacity



class SequentialReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, device = 'auto', seed = 123456) -> None:
        super().__init__(capacity, device, seed)

        self.sample_helper_func = lambda x: torch.from_numpy(np.stack(x, 0)).float().to(self.device)


    def sample(self, batch_size):
        tmp_buffer = tuple(map(self.sample_helper_func, zip(*self.buffer)))
        n_envs = tmp_buffer[0].size(1)
        sampled_env_indices = np.random.permutation(n_envs)[:batch_size]
        return tuple(map(lambda x: x[:, sampled_env_indices], tmp_buffer))

    
    def generate_dataset(self, batch_size = None):
        tmp_buffer = tuple(map(self.sample_helper_func, zip(*self.buffer)))
        n_envs = tmp_buffer[0].size(1)
        env_indices = np.random.permutation(n_envs)
        if batch_size is None:
            batch_size = self.size
        start_idx = 0
        while start_idx < self.size:
            yield tuple(map(lambda x: x[:, env_indices[start_idx:start_idx + batch_size]], tmp_buffer))
            start_idx += batch_size

    


if __name__ == '__main__':
    buf = ReplayBuffer(10)
    for _ in range(6):
        buf.add((np.random.randn(3, 4), np.random.randn(3, 5)))
    print([b.size() for b in buf.sample(2)])
    for batch in buf.generate_dataset():
        print([b.size() for b in batch])

    buf = SequentialReplayBuffer(10)
    for _ in range(8):
        buf.add((np.random.randn(3, 4), np.random.randn(3, 5)))
    print([b.size() for b in buf.sample(2)])
    for batch in buf.generate_dataset():
        print([b.size() for b in batch])