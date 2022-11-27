import torch
import torch.nn as nn


def create_mlp(in_dim, out_dim, hidden_dim_list = [256], act_fn = None, last_act_fn = None):
    layer_dim_list = [in_dim] + hidden_dim_list
    layers = []

    for i in range(len(layer_dim_list) - 1):
        layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
        if act_fn is not None:
            layers.append(act_fn())

    layers.append(nn.Linear(layer_dim_list[-1], out_dim))
    if last_act_fn is not None:
        layers.append(last_act_fn())

    return nn.Sequential(*layers)


def soft_update(target, source, tau):
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.copy_(source_param * tau + target_param * (1.0 - tau))


def hard_update(target, source):
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.copy_(source_param)


'''
    Predict the continuous action space.
'''
class GaussianPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_list = [256], min_logstd = -20, max_logstd = 2, mu_act_fn = 'tanh', action_space = None) -> None:
        super().__init__()

        self.main = create_mlp(in_dim, out_dim * 2, hidden_dim_list, nn.ReLU, None)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.mu_act_fn = mu_act_fn

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = 0.5 * (action_space.high - action_space.low)
            self.action_bias = 0.5 * (action_space.high + action_space.low)


    def forward(self, x):
        p = self.main(x)
        mu, logstd = torch.chunk(p, 2, -1)
        std = torch.clamp(logstd, self.min_logstd, self.max_logstd).exp()
        dist = torch.distributions.Normal(mu, std)
        a_t_before_act_fn = dist.rsample() # Reparameterization trick

        if self.mu_act_fn:
            # Enforcing action bound
            if self.mu_act_fn == 'tanh':
                a_t = torch.tanh(a_t_before_act_fn)
                log_prob = torch.sum(dist.log_prob(a_t_before_act_fn) - torch.log(self.action_scale * (1 - a_t.pow(2)) + 1e-4), -1)
                mu = torch.tanh(mu)
            else:
                raise NotImplementedError
        else:
            a_t = a_t_before_act_fn
            log_prob = torch.sum(dist.log_prob(a_t), -1)

        action = self.action_scale * a_t + self.action_bias        
        return action, log_prob, mu


if __name__ == '__main__':
    print(create_mlp(4, 2, [64], act_fn = nn.ReLU, last_act_fn = nn.Tanh))