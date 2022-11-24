import numpy as np
import torch
import torch.nn as nn
from base_policy import BasePolicy
from utils import np_to_tensor, tensor_to_np
from network import GaussianPolicy, create_mlp
from buffers import ReplayBuffer
from logger import Logger
from multienv import SubprocVecEnv

'''
    https://github.com/jannerm/mbpo
    https://github.com/jxu43/replication-mbpo
    https://github.com/pranz24/pytorch-soft-actor-critic
'''
class SoftQNet(nn.Module):
    def __init__(self, in_dim, hidden_dim_list = [256], act_fn = nn.ReLU) -> None:
        super().__init__()

        self.q1_net = create_mlp(in_dim, 1, hidden_dim_list, act_fn, None)
        self.q2_net = create_mlp(in_dim, 1, hidden_dim_list, act_fn, None)


    def forward(self, o, a):
        oa = torch.cat([o, a], -1)
        q1 = self.q1_net(oa).squeeze(-1)
        q2 = self.q2_net(oa).squeeze(-1)
        return q1, q2



class SACPolicy(BasePolicy):
    def __init__(self, 
        obs_space, 
        act_space, 
        hidden_dims, 
        alpha = 0.2, 
        gamma = 0.99, 
        tau = 0.005,
        actor_lr = 0.0003,
        critic_lr = 0.0003,
        lr_scheduler = None,
        device = 'auto'
    ) -> None:
        super().__init__(obs_space, act_space, lr_scheduler, device)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianPolicy(self.obs_dim, self.act_dim, hidden_dims)
        self.actor.apply(self.init_weights)
        self.critic = SoftQNet(self.obs_dim + self.act_dim, hidden_dims)
        self.critic_target = SoftQNet(self.obs_dim + self.act_dim, hidden_dims)
        self.critic.apply(self.init_weights)

        self.hard_update(self.critic_target, self.critic)
        self = self.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)


    def predict(self, obs, state = None):
        return self.actor.forward(obs)


    def learn(self, batch):
        obs_batch, next_obs_batch, action_batch, reward_batch, done_batch = batch

        with torch.no_grad():
            next_action, next_action_logprob, _ = self.predict(next_obs_batch)
            target_q1, target_q2 = self.critic_target(next_obs_batch, next_action)
            target_V = torch.min(target_q1, target_q2) - self.alpha * next_action_logprob
            target_Q = reward_batch + (1.0 - done_batch) * self.gamma * target_V

        q1, q2 = self.critic(obs_batch, action_batch)
        q_loss = 0.5 * ((q1 - target_Q).pow(2) + (q2 - target_Q).pow(2)).mean()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        new_action, new_action_logprob, _ = self.predict(obs_batch)
        q1_new, q2_new = self.critic(obs_batch, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_action_logprob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_target, self.critic, self.tau)

        return {
            'policy_loss' : policy_loss.item(),
            'q_loss' : q_loss.item()
        }


    def get_policy_dict(self):
        return {
            'actor' : self.actor,
            'critic' : self.critic,
            'critic_target' : self.critic_target,
            'actor_optimizer' : self.actor_optimizer,
            'critic_optimizer' : self.critic_optimizer
        }



def sac_trainer(
    env : SubprocVecEnv, 
    model : SACPolicy, 
    buffer : ReplayBuffer, 
    batch_size : int, 
    epoch : int, 
    updates_per_step : int,
    log_path = './sac/',
    eval_frequency = 5,
):
    total_updates = 0
    total_steps = 0
    log = Logger(log_path)

    for e in np.arange(epoch) + 1:
        obs = env.reset()
        episode_rewards_list, episode_steps_list = [], []
        for _ in range(env._max_episode_steps):
            with torch.no_grad():
                action, _, _ = model.predict(obs)
                action = tensor_to_np(action)

            if buffer.size > batch_size:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    log_dict = model.update(batch)
                    log.add(total_updates, log_dict, 'Train')
                    total_updates += 1

            next_obs, rewards, dones, infos = env.step(action)
            total_steps += 1
            buffer.add((obs, next_obs, action, rewards, dones))
            obs = next_obs.copy()
            for info in infos:
                if info['terminate']:
                    episode_rewards_list.append(info['episode_reward'])
                    episode_steps_list.append(info['episode_step'])
        print('Epoch {:d} Sample: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(e, np.mean(episode_rewards_list), np.std(episode_rewards_list), np.mean(episode_steps_list), np.std(episode_steps_list)))


        if e % eval_frequency == 0:
            obs = env.reset()
            episode_rewards_list, episode_steps_list = [], []
            for _ in range(env._max_episode_steps):
                with torch.no_grad():
                    _, _, action = model.predict(obs)
                    action = tensor_to_np(action)
                next_obs, rewards, dones, infos = env.step(action)
                obs = next_obs.copy()
                for info in infos:
                    if info['terminate']:
                        episode_rewards_list.append(info['episode_reward'])
                        episode_steps_list.append(info['episode_step'])            
            print('Epoch {:d} Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(e, np.mean(episode_rewards_list), np.std(episode_rewards_list), np.mean(episode_steps_list), np.std(episode_steps_list)))




if __name__ == '__main__':
    from sac import SACPolicy
    from multienv import make_mp_envs
    from buffers import ReplayBuffer
    from collector import Collector
    from base_trainer import OffPolicyTrainer
    import gym

    env_id = 'HalfCheetah-v4'
    env = make_mp_envs(env_id, n_envs = 4)
    eval_env = gym.make(env_id)
    buffer = ReplayBuffer(50000)
    model = SACPolicy(
        env.observation_space, 
        env.action_space, 
        hidden_dims = [256, 256],
    )
    collector = Collector(env, buffer, model)
    trainer = OffPolicyTrainer(
        model = model,
        collector = collector,
        epochs = 60,
        batch_size = 128,
        log_path = './sac/',
        best_model_path = './sac/best.pkl',
        eval_frequency = 5,
        steps_per_collect = 1000,
        update_per_step = 1
    )
    trainer.learn()
    model.save('./sac/model.pkl')
    collector.record_video(eval_env, './sac/curr.mp4')
    model.load('./sac/best.pkl')
    collector.record_video(eval_env, './sac/best.mp4')