import torch
import numpy as np
from base_policy import BasePolicy
from buffers import ReplayBuffer
from multienv import SubprocVecEnv
from utils import tensor_to_np, np_to_tensor
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Collector(object):
    def __init__(self, 
        env : SubprocVecEnv, 
        buffer : ReplayBuffer, 
        model : BasePolicy = None,
        preprocess_fn = None
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.model = model
        self.preprocess_fn = preprocess_fn


    def collect(self, n_steps = None, is_training = True, reset_buffer = False):
        obs = self.env.reset()
        episode_rewards_list, episode_steps_list = [], []

        if reset_buffer:
            self.buffer.reset()

        if n_steps is None:
            n_steps = self.buffer.capacity 

        for _ in range(n_steps):
            if self.model is None:
                action = self.env.sample_actions()
            else:
                with torch.no_grad():
                    if self.preprocess_fn:
                        obs = self.preprocess_fn(obs)
                    obs_t = np_to_tensor(obs, self.model.device).float()
                    a_t, _, mean = self.model.predict(obs_t)
                    action = tensor_to_np(a_t if is_training else mean)
            
            next_obs, rewards, dones, infos = self.env.step(action)
            if is_training:
                self.buffer.add((obs, next_obs, action, rewards, dones))
            obs = next_obs.copy()

            for info in infos:
                if info['terminate']:
                    episode_rewards_list.append(info['episode_reward'])
                    episode_steps_list.append(info['episode_step'])

        log_info = {
            'Sample Rewards Mean' : np.mean(episode_rewards_list),
            'Sample Rewards Std' : np.std(episode_rewards_list),
            'Sample Steps Mean' : np.mean(episode_steps_list),
            'Sample Steps Std' : np.std(episode_steps_list),
            'n_steps' : n_steps * self.env.n_envs
        }

        return log_info


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
                    if self.preprocess_fn:
                        obs = self.preprocess_fn(obs)
                    obs_t = np_to_tensor(obs, self.model.device).float()
                    _, _, mean = self.model.predict(obs_t)
                    action = tensor_to_np(mean)

            obs, reward, done, _ = eval_env.step(action)
            video_recorder.capture_frame()
            episode_reward += reward
            episode_step += 1
            if done:
                break

        print('Reward = %.3f | Step = %d' % (episode_reward, episode_step))
        video_recorder.close()
        video_recorder.enabled = False


if __name__ == '__main__':
    from multienv import make_mp_envs
    from buffers import ReplayBuffer
    import gym
    env_id = 'Humanoid-v4'
    env = make_mp_envs(env_id, n_envs = 4)
    eval_env = gym.make(env_id)
    buffer = ReplayBuffer(50000)
    collector = Collector(env, buffer)
    log_info = collector.collect(1000)
    print(log_info)
    # collector.record_video(eval_env, './test.mp4')