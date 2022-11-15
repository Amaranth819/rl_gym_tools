import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from utils import get_device, np_to_tensor, tensor_to_np
from base_policy import BasePolicy
from buffers import RolloutBuffer
from multienv import SubprocVecEnv
from logger import Logger


class BaseTrainer(object):
    def __init__(self, 
        env : SubprocVecEnv, 
        buffer : RolloutBuffer, 
        model : BasePolicy, 
        optimizer : torch.optim.Optimizer, 
        lr_scheduler : torch.optim.lr_scheduler = None, 
        device : str = 'auto',
    ) -> None:
        self.device = get_device(device)
        self.env = env
        self.buffer = buffer
        self.buffer.device = self.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def collect(self, model_prediction = True, deterministic = False):
        obs = self.env.reset()
        self.buffer.reset()

        episode_rewards_list, episode_steps_list = [], []
        
        while not self.buffer.isfull:
            if model_prediction:
                with torch.no_grad():
                    obs_ts = np_to_tensor(obs, self.device)
                    action_ts = self.model.predict(obs_ts, deterministic = deterministic)
                action = tensor_to_np(action_ts)
            else:
                action = self.env.sample_actions()

            next_obs, rewards, dones, infos = self.env.step(action)
            self.buffer.add(obs, next_obs, action, rewards, dones)

            for info in infos:
                if 'total_rewards' in info.keys():
                    episode_rewards_list.append(info['total_rewards'])
                    episode_steps_list.append(info['total_steps'])
            obs = next_obs

        return episode_rewards_list, episode_steps_list

    
    def train(self, batch_size, sequential = False):
        log_info = defaultdict(lambda: [])

        # Sampling trajectory
        episode_rewards_list, episode_steps_list = self.collect(False, False)
        log_info['Sample Rewards Mean'] = np.mean(episode_rewards_list)
        log_info['Sample Steps Mean'] = np.mean(episode_steps_list)

        # Training
        for batch in self.buffer.generate_data(batch_size, sequential):
            batch_training_log_info = self._train(batch)

            for tag, val in batch_training_log_info.items():
                log_info[tag].append(val)

        # Summary
        for tag, val_list in log_info.items():
            log_info[tag] = np.mean(val_list)

        return log_info


    def _train(self, batch, clip_grad_max_norm = None):
        loss, batch_training_log_info = self.model.update(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = clip_grad_max_norm)
        self.optimizer.step()
        return batch_training_log_info


    def learn(self, epochs, batch_size, eval_frequency = 10, log_path = 'log/', best_model_path = None):
        if epochs <= 0:
            print('No training task!')
            return

        log = Logger(log_path)

        # Eval before training
        if eval_frequency is not None:
            eval_episode_rewards_list, eval_episode_steps_list = self.collect(False, True)
            eval_log_info = {
                'Sample Rewards Mean' : np.mean(eval_episode_rewards_list),
                'Sample Rewards Std' : np.std(eval_episode_rewards_list),
                'Sample Steps Mean' : np.mean(eval_episode_steps_list),
                'Sample Steps Std' : np.std(eval_episode_steps_list)
            }
            best_eval_rewards_mean = eval_log_info['Sample Rewards Mean']
            log.add(0, eval_log_info, 'Eval/')
            print('Epoch 0 Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(*eval_log_info.values()))

        # Training
        for e in np.arange(epochs) + 1:
            training_info = self.train(batch_size)

            if log is not None:
                log.add(e, training_info, 'Train/')

            print('####################')
            print('# Epoch: %d' % e)
            for tag, scalar_val in training_info.items():
                print('# %s: %.5f' % (tag, scalar_val))
            print('####################\n')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Evaluate every certain epochs
            if eval_frequency is not None and e % eval_frequency == 0:
                eval_episode_rewards_list, eval_episode_steps_list = self.collect(False, True)
                eval_log_info = {
                    'Sample Rewards Mean' : np.mean(eval_episode_rewards_list),
                    'Sample Rewards Std' : np.std(eval_episode_rewards_list),
                    'Sample Steps Mean' : np.mean(eval_episode_steps_list),
                    'Sample Steps Std' : np.std(eval_episode_steps_list)
                }
                log.add(e, eval_log_info, 'Eval/')
                print('Epoch {:d} Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(e, *eval_log_info.values()))

                if eval_log_info['Sample Rewards Mean'] > best_eval_rewards_mean:
                    print('Get a better model!\n')
                    best_eval_rewards_mean = eval_log_info['Sample Rewards Mean']
                    if best_model_path is not None:
                        self.save(best_model_path)
                else:
                    print('Don\'t get a better model!\n')

    
    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, path)


    def load(self, pkl_path):
        state_dict = torch.load(pkl_path, map_location = self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        print('Load from %s successfully!' % pkl_path)


    '''
        Visualize gradient flow
    '''
    def visualize_gradient_flow(self):
        def plot_grad_flow(named_parameters):
            '''Plots the gradients flowing through different layers in the net during training.
            Can be used for checking for possible gradient vanishing / exploding problems.
            
            Usage: Plug this function in Trainer class after loss.backwards() as 
            "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
            ave_grads = []
            max_grads= []
            layers = []
            for n, p in named_parameters:
                if(p.requires_grad) and ("bias" not in n):
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())

            plt.bar(np.arange(len(max_grads)), max_grads, alpha = 1, lw = 1, color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha = 1, lw = 1, color="b")
            plt.hlines(0, 0, len(ave_grads) + 1, lw = 2, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(left = 0, right = len(ave_grads))
            plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
            plt.xlabel("Layers")
            plt.ylabel("Average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            plt.legend([Line2D([0], [0], color="c", lw = 4),
                        Line2D([0], [0], color="b", lw = 4),
                        Line2D([0], [0], color="k", lw = 4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
            plt.tight_layout()
            plt.savefig('GradientFlow.png')

        self.collect(False, False)
        self.train(batch_size = None, sequential = False)
        plot_grad_flow(self.model.cpu().named_parameters())