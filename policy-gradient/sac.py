import numpy as np
import torch
import torch.nn as nn
from base_trainer import BaseTrainer
from base_policy import BasePolicy
from utils import np_to_tensor, tensor_to_np
from network import GaussianPolicy, create_mlp
from collector import Collector

'''
    https://github.com/jannerm/mbpo
    https://github.com/jxu43/replication-mbpo
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


def soft_update(target_critic, source_critic, tau):
    with torch.no_grad():
        for source_param, target_param in zip(source_critic.parameters(), target_critic.parameters()):
            target_param.copy_(source_param * tau + target_param * (1.0 - tau))


def hard_update(target_critic, source_critic):
    with torch.no_grad():
        for source_param, target_param in zip(source_critic.parameters(), target_critic.parameters()):
            target_param.copy_(source_param)


class SACPolicy(BasePolicy):
    def __init__(self, obs_space, act_space, hidden_dim, alpha = 0.1, gamma = 0.99, tau = 0.005, device = 'auto') -> None:
        super().__init__(obs_space, act_space, False, device)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianPolicy(self.obs_dim, self.act_dim, [hidden_dim], nn.SiLU)
        self.critic = SoftQNet(self.obs_dim + self.act_dim, [hidden_dim])
        self.critic_target = SoftQNet(self.obs_dim + self.act_dim, [hidden_dim])

        hard_update(self.critic_target, self.critic)
        self = self.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.0003, betas = [0.9, 0.999])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.0001, betas = [0.9, 0.999])
    

    def predict(self, obs, deterministic = False):
        if type(obs) == np.ndarray:
            obs = np_to_tensor(obs, self.device)
        dist, (action_dist_mu, action_dist_std) = self.actor(obs.float())
        action_ts = action_dist_mu if deterministic else dist.sample()
        action_logprob_ts = dist.log_prob(action_ts)
        return action_ts, action_logprob_ts, (action_dist_mu, action_dist_std)


    def update(self, batch):
        obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, _, _ = batch

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

        soft_update(self.critic_target, self.critic, self.tau)

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


if __name__ == '__main__':
    from multienv import make_mp_envs
    from buffers import RolloutBuffer
    import gym
    env = make_mp_envs('HalfCheetah-v4', n_envs = 8)
    eval_env = gym.make('HalfCheetah-v4')
    buffer = RolloutBuffer(env.observation_space, env.action_space, env._max_episode_steps, n_envs = env.n_envs)
    model = SACPolicy(
        env.observation_space, 
        env.action_space,
        hidden_dim = 256
    )
    collector = Collector(env, buffer, model)
    trainer = BaseTrainer(
        collector,
        model, 
        lr_scheduler_dict = None,
        epochs = 50,
        batch_size = 128,
        log_path = './sac/',
        best_model_path = './sac/best.pkl',
        eval_frequency = 5,
        n_sample_steps = None,
        reset_buffer_every_collect = True
    )
    trainer.learn()
    model.save('./sac/model.pkl')
    collector.record_video(eval_env, './sac/curr.mp4')
    model.load('./sac/best.pkl')
    collector.record_video(eval_env, './sac/best.mp4')