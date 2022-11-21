import torch
import numpy as np
from base_policy import BasePolicy
from buffers import RolloutBuffer, BaseBuffer
from multienv import SubprocVecEnv
from utils import tensor_to_np
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Collector(object):
    def __init__(self, 
        env : SubprocVecEnv, 
        buffer : BaseBuffer, 
        model : BasePolicy = None,
        preprocess_fn = None
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.model = model
        self.preprocess_fn = preprocess_fn


    def collect(self, n_steps = None, deterministic = False, reset_buffer = True):
        if reset_buffer:
            self.buffer.reset()

        episode_rewards_list, episode_steps_list = [], []

        if n_steps is None:
            n_steps = self.buffer.buffer_size 

        obs = self.env.reset()
        for _ in range(n_steps):
            if self.model is None:
                action = self.env.sample_actions()
            else:
                with torch.no_grad():
                    if self.preprocess_fn:
                        obs = self.preprocess_fn(obs)
                    action_ts, _, _ = self.model.predict(obs, deterministic = deterministic)
                    action = tensor_to_np(action_ts)
            
            next_obs, rewards, dones, infos = self.env.step(action)
            self.buffer.add(obs, next_obs, action, rewards, dones)

            for info in infos:
                if info['terminate']:
                    episode_rewards_list.append(info['episode_reward'])
                    episode_steps_list.append(info['episode_step'])
            obs = next_obs.copy()

        log_info = {
            'Sample Rewards Mean' : np.mean(episode_rewards_list),
            'Sample Rewards Std' : np.std(episode_rewards_list),
            'Sample Steps Mean' : np.mean(episode_steps_list),
            'Sample Steps Std' : np.std(episode_steps_list)
        }

        return log_info


    def generate_data(self, batch_size = None, sequential = False):
        return self.buffer.generate_data(batch_size, sequential)


    def sample(self, batch_size = None, sequential = False):
        return next(iter(self.buffer.generate_data(batch_size, sequential)))


    def record_video(self, eval_env, video_path):
        obs = eval_env.reset()
        episode_reward, episode_step = 0, 0
        video_recorder = VideoRecorder(eval_env, video_path, enabled = True)
        max_episode_steps = eval_env._max_episode_steps if hasattr(eval_env, '_max_episode_steps') else 1000

        for _ in range(max_episode_steps):
            if self.model is None:
                action = eval_env.action_space.sample()
            else:
                with torch.no_grad():
                    # obs = obs[None]
                    if self.preprocess_fn:
                        obs = self.preprocess_fn(obs)
                    action_ts, _, _ = self.model.predict(obs, deterministic = True)
                    action = tensor_to_np(action_ts)

            next_obs, reward, done, _ = eval_env.step(action)

            video_recorder.capture_frame()
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            if done:
                break

        print('Reward = %.3f | Step = %d' % (episode_reward, episode_step))
        video_recorder.close()
        video_recorder.enabled = False


if __name__ == '__main__':
    from multienv import make_mp_envs
    from buffers import RolloutBuffer
    import gym
    env_id = 'Humanoid-v4'
    env = make_mp_envs(env_id, n_envs = 4)
    eval_env = gym.make(env_id)
    buffer = RolloutBuffer(
        env.observation_space,
        env.action_space,
        env._max_episode_steps,
        4
    )
    collector = Collector(env, buffer)
    episode_rewards_list, episode_steps_list = collector.collect(1000)
    print(episode_rewards_list)
    print(episode_steps_list)
    # collector.record_video(eval_env, './test.mp4')