import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import get_device, get_space_shape

class BasePolicy(nn.Module):
    def __init__(self, obs_space, act_space, is_recurrent = False, device = 'auto') -> None:
        super().__init__()

        self.obs_shape = get_space_shape(obs_space)
        self.obs_dim = np.prod(self.obs_shape)
        self.act_shape = get_space_shape(act_space)
        self.act_dim = np.prod(self.act_shape)
        self.device = get_device(device)
        self.is_recurrent = is_recurrent


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # For FC
                torch.nn.init.normal_(m.weight.data, mean = 0, std = 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                # For recurrent network
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)


    def predict(self, obs, deterministic = False):
        raise NotImplementedError


    def update(self, batch):
        raise NotImplementedError


    def get_policy_dict(self):
        # dict( name : (nn.Module or optimizer) )
        raise NotImplementedError


    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {}
        for name, component in self.get_policy_dict().items():
            state_dict[name] = component.state_dict()
        torch.save(state_dict, path)


    def load(self, pkl_path):
        state_dict = torch.load(pkl_path)
        for name, component in self.get_policy_dict().items():
            component.load_state_dict(state_dict[name])
        print('Load from %s successfully!' % pkl_path)


    '''
        Visualize gradient flow
    '''
    def visualize_gradient_flow(self, batch):
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
                    if p.grad is None:
                        ave_grads.append(0)
                        max_grads.append(0)
                    else:
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

        self.update((b.to(self.device) for b in batch))
        plot_grad_flow(self.cpu().named_parameters())