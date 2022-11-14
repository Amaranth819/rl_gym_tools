import gym
import numpy as np
import pickle
import cloudpickle
import time
from multiprocessing import Pipe, Process

'''
    Reference: 
    1. https://squadrick.dev/journal/efficient-multi-gym-environments.html
    2. https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/subproc_vec_env.html#SubprocVecEnv
'''
class CloudpickleWrapper(object):
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		self.x = pickle.loads(ob)
	
	def __call__(self):
		return self.x()


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    total_rewards = 0
    total_steps = 0

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            total_rewards += reward
            total_steps += 1
            if done:
                obs = env.reset()
                info['total_rewards'] = total_rewards
                info['total_steps'] = total_steps
                total_rewards = 0
                total_steps = 0
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs = env.reset()
            total_rewards = 0
            total_steps = 0
            remote.send(obs)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'render':
            remote.send(env.render())
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError("`{}` is not implemented in the worker.".format(cmd))


class SubprocVecEnv(object):
    def __init__(self, env_fn_list) -> None:
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fn_list)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        self.ps = []
        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fn_list):
            process = Process(target = worker, args = (wrk, rem, CloudpickleWrapper(fn)))
            process.daemon = True
            process.start()
            self.ps.append(process)
            wrk.close()

        self.remotes[0].send(('get_spaces', None))
        self.single_observation_space, self.single_action_space = self.remotes[0].recv()


    def sample_actions(self):
        return np.stack([self.single_action_space.sample() for _ in range(self.n_envs)])


    def step_async(self, actions):
        if self.waiting:
            raise TypeError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    
    def step_wait(self):
        if not self.waiting:
            raise TypeError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), np.stack(infos)


    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def make_mp_envs(env_id = '', env_fn = None, n_envs = 4):
    def fn():
        env = gym.make(env_id)
        return env
    return SubprocVecEnv([env_fn if env_fn else fn for _ in range(n_envs)])




# class VecEnv(object):
#     def __init__(self, env_id, n_envs) -> None:
#         self.envs = [gym.make(env_id) for _ in range(n_envs)]


#     def reset(self):
#         return np.stack([env.reset() for env in self.envs])


#     def step(self, actions):
#         results = [env.step(a) for env, a in zip(self.envs, actions)]
#         obs, rewards, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rewards), np.stack(dones), {}


if __name__ == '__main__':
    from halfcheetah import HalfCheetahEnv_RandomDynamics
    gym.envs.register(
        id='HalfCheetahRandomDynamics-v0',
        entry_point='halfcheetah:HalfCheetahEnv_RandomDynamics',
        max_episode_steps=1000,
    )
    # env_fn = lambda: HalfCheetahEnv_RandomDynamics()
    env_fn = None
    envs = make_mp_envs('HalfCheetahRandomDynamics-v0', env_fn = env_fn, n_envs = 4)
    # envs = VecEnv('Ant-v4', 20)
    act_dim = envs.single_action_space.shape[0]
    envs.reset()
    curr = time.time()
    rewards = []
    steps = []
    for e in range(1002):
        _, _, dones, infos = envs.step(envs.sample_actions())
        # print(e, np.all(dones))
        for info in infos:
            if 'total_rewards' in info.keys():
                rewards.append(info['total_rewards'])
                steps.append(info['total_steps'])
    print(time.time() - curr)
    print(rewards, np.mean(rewards))
    print(steps, np.mean(steps))