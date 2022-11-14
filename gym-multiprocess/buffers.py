import torch
import numpy as np
import functools
from utils import get_device, get_space_shape



'''
    Base buffer class
'''
class BaseBuffer(object):
    def __init__(self, obs_space, act_space, buffer_size, n_envs = 1, device = 'auto') -> None:
        self.obs_space = obs_space
        self.act_space = act_space
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = get_device(device)

        self.obs_shape = get_space_shape(self.obs_space)
        self.act_shape = get_space_shape(self.act_space)
        self.pos = 0


    def add(self):
        raise NotImplementedError


    def reset(self):
        self.pos = 0


    def flatten(self, arr):
        shape = arr.shape
        return np.swapaxes(arr, 0, 1).reshape(shape[0] * shape[1], *shape[2:])


    def generate_data(self, batch_size = None, sequential = False):
        buffer_list = self._get_buffer_list()
        if sequential:
            total_size = self.n_envs
            indexing_func = lambda x, inds: x[:, inds]
        else:
            buffer_list = tuple(map(lambda x: self.flatten(x), buffer_list))
            total_size = self.n_envs * self.pos
            indexing_func = lambda x, inds: x[inds]
        
        inds = np.random.permutation(total_size)
        if batch_size is None:
            batch_size = total_size

        start_idx = 0
        while start_idx < total_size:
            yield map(functools.partial(indexing_func, inds = inds[start_idx:start_idx + batch_size]), buffer_list)
            start_idx += batch_size



    def _get_buffer_list(self):
        raise NotImplementedError


    @property
    def size(self):
        return self.pos


    @property
    def isfull(self):
        return self.pos >= self.buffer_size



'''
    Replay buffer for off-policy algorithms
'''
class ReplayBuffer(BaseBuffer):
    def __init__(self, obs_space, act_space, buffer_size, n_envs = 1, device = 'auto', overwrite = False) -> None:
        super().__init__(obs_space, act_space, buffer_size, n_envs, device) 

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype = self.obs_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype = self.obs_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs) + self.act_shape, dtype = self.act_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.overwrite = overwrite


    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = np.copy(obs)
        self.next_observations[self.pos] = np.copy(next_obs)
        self.actions[self.pos] = np.copy(action)
        self.rewards[self.pos] = np.copy(reward)
        self.dones[self.pos] = np.copy(done)
        self.pos = (self.pos + 1) % self.buffer_size if self.overwrite else self.pos + 1


    def _get_buffer_list(self):
        return self.observations, self.next_observations, self.actions, self.rewards, self.dones




'''
    Rollout buffer for on-policy algorithms
'''
class RolloutBuffer(BaseBuffer):
    def __init__(self, obs_space, act_space, buffer_size, n_envs = 1, device = 'auto') -> None:
        super().__init__(obs_space, act_space, buffer_size, n_envs, device)

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype = self.obs_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype = self.obs_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs) + self.act_shape, dtype = self.act_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype = np.float32)


    def add(self, obs, next_obs, action, reward, done, value = None, log_prob = None):
        self.observations[self.pos] = np.copy(obs)
        self.next_observations[self.pos] = np.copy(next_obs)
        self.actions[self.pos] = np.copy(action)
        self.rewards[self.pos] = np.copy(reward)
        self.dones[self.pos] = np.copy(done)
        if value is not None:
            self.values[self.pos] = np.copy(value)
        if log_prob is not None:
            self.log_probs[self.pos] = np.copy(log_prob)
        self.pos += 1


    def compute_returns_and_advantage(self, last_values, last_dones, gae_lambda, gamma):
        last_gae_lam = 0
        for s in reversed(range(self.pos)):
            if s == self.pos - 1:
                next_not_done = 1.0 - last_dones
                next_values = last_values
            else:
                next_not_done = 1.0 - self.dones[s + 1]
                next_values = self.values[s + 1]
            
            delta = self.rewards[s] + gamma * next_values * next_not_done - self.values[s]
            last_gae_lam = delta + gamma * gae_lambda * next_not_done * last_gae_lam
            self.advantages[s] = last_gae_lam

        self.returns = self.advantages + self.values


    def _get_buffer_list(self):
        return self.observations, self.next_observations, self.actions, self.rewards, self.dones, self.values, self.log_probs



if __name__ == '__main__':
    from multienv import make_mp_envs
    n = 20
    n_envs = 2
    env = make_mp_envs('Humanoid-v4', n_envs = n_envs)
    obs = env.reset()
    buffer = ReplayBuffer(env.single_observation_space, env.single_action_space, n, n_envs)

    for e in range(n):
        action = env.sample_actions()
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, next_obs, action, reward, done)
        obs = next_obs
    
    # buffer.compute_returns_and_advantage(np.zeros(1), done, 1, 1)    
    # for e in range(n):
    #     print(buffer.rewards[e], buffer.values[e], buffer.advantages[e], buffer.dones[e])
    
    for obs, next_obs, action, reward, done in buffer.generate_data(batch_size = 2, sequential = True):
        print(obs.shape)